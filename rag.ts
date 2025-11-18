import fs from "fs";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";

// Configuration
const DEFAULT_EMBEDDING_MODEL = "nomic-embed-text";
const DEFAULT_CHAT_MODEL = process.env.OLLAMA_MODEL || "deepseek-v3.1:671b-cloud";
const DEFAULT_OLLAMA_URL = "http://localhost:11434";
const DEFAULT_CHUNK_SIZE = 1000;
const DEFAULT_CHUNK_OVERLAP = 200;
const DEFAULT_TOP_K = 9;
const MIN_CONTENT_LENGTH = 20; // Minimum characters for a document to be considered valid
const MIN_WORD_COUNT = 5; // Minimum words for a document to be considered valid

// Cosine similarity function
function cosine(a: number[], b: number[]): number {
    if (!Array.isArray(a) || !Array.isArray(b)) return 0;
    const len = Math.min(a.length, b.length);
    if (len === 0) return 0;

    let dot = 0;
    let na = 0;
    let nb = 0;

    for (let i = 0; i < len; i++) {
        const ai = Number(a[i]) || 0;
        const bi = Number(b[i]) || 0;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }

    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Helper function to count words in text
function countWords(text: string): number {
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
}

// $ Ingest PDF and create embeddings
async function ingestPDF(pdfPath: string, outputPath: string): Promise<void> {
    try {
        console.log(`Loading PDF file: ${pdfPath}`);
        const loader = new PDFLoader(pdfPath, {
            splitPages: true,
            // These options help focus on text content and ignore images
            parsedItemSeparator: " ",
        });
        const data = await loader.load();

        console.log(`Loaded ${data.length} documents from PDF`);

        // Log more information about the loaded documents
        if (data.length > 0) {
            console.log(`First document pageContent length: ${data[0].pageContent?.length || 0}`);
            console.log(`First document word count: ${countWords(data[0].pageContent || "")}`);
            console.log(`First document metadata:`, JSON.stringify(data[0].metadata, null, 2));

            // Show first few characters of content
            if (data[0].pageContent) {
                console.log(`First 100 characters of content: "${data[0].pageContent.substring(0, 100)}"`);
            }

            // Check if there's any document with substantial content
            const docsWithContent = data.filter(doc => {
                const content = doc.pageContent || "";
                return content.trim().length > MIN_CONTENT_LENGTH && countWords(content) >= MIN_WORD_COUNT;
            });
            console.log(`Documents with more than ${MIN_CONTENT_LENGTH} characters and ${MIN_WORD_COUNT} words: ${docsWithContent.length}`);

            if (docsWithContent.length > 0) {
                console.log(`Best document has ${docsWithContent[0].pageContent.length} characters and ${countWords(docsWithContent[0].pageContent)} words`);
                console.log(`Content preview: "${docsWithContent[0].pageContent.substring(0, 200)}"`);
            }
        }

        // Filter out documents with minimal content (focus on text, ignore images)
        const filteredDocs = data.filter(doc => {
            const content = doc.pageContent || "";
            const trimmedContent = content.trim();
            return trimmedContent.length > MIN_CONTENT_LENGTH && countWords(trimmedContent) >= MIN_WORD_COUNT;
        });

        console.log(`Filtered to ${filteredDocs.length} documents with substantial text content`);

        if (filteredDocs.length === 0) {
            console.warn("No documents with substantial text content found. This might be because the PDF is image-based or has no extractable text.");
            // Save an empty array and exit
            fs.writeFileSync(outputPath, JSON.stringify([], null, 2), "utf-8");
            console.log(`Saved empty embeddings array to ${outputPath}`);
            return;
        }

        console.log("Splitting text into chunks...");
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: DEFAULT_CHUNK_SIZE,
            chunkOverlap: DEFAULT_CHUNK_OVERLAP,
        });

        const docs = await textSplitter.splitDocuments(filteredDocs);
        console.log(`Split into ${docs.length} chunks`);

        if (docs.length === 0) {
            console.warn("No chunks were created. This might be because the PDF content is not suitable for chunking.");
            // Save an empty array and exit
            fs.writeFileSync(outputPath, JSON.stringify([], null, 2), "utf-8");
            console.log(`Saved empty embeddings array to ${outputPath}`);
            return;
        }

        // Clean the documents properly (focus on text content)
        const cleanedDocs = docs.map((d, i) => {
            // More aggressive cleaning to focus on actual text content
            let cleanContent = d.pageContent
                .replace(/\s+/g, " ") // Replace multiple whitespace with single space
                .replace(/\n+/g, " ") // Replace multiple newlines with single space
                .trim(); // Remove leading/trailing whitespace

            // Remove any remaining control characters except common ones
            cleanContent = cleanContent.replace(/[^\x20-\x7E\x0A\x0D\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/g, "");

            return {
                ...d,
                pageContent: cleanContent,
                metadata: { ...d.metadata, chunkIndex: i, source: d.metadata?.source ?? "" },
            };
        });

        // Filter out any documents that became empty after cleaning
        const finalDocs = cleanedDocs.filter(doc => doc.pageContent.length > 0 && countWords(doc.pageContent) >= MIN_WORD_COUNT);
        console.log(`After cleaning, ${finalDocs.length} documents remain with sufficient text content`);

        if (finalDocs.length === 0) {
            console.warn("No valid text content remaining after cleaning.");
            // Save an empty array and exit
            fs.writeFileSync(outputPath, JSON.stringify([], null, 2), "utf-8");
            console.log(`Saved empty embeddings array to ${outputPath}`);
            return;
        }

        console.log("Generating embeddings...");
        const embeddings = new OllamaEmbeddings({
            model: DEFAULT_EMBEDDING_MODEL,
            baseUrl: DEFAULT_OLLAMA_URL,
        });

        const vectors = await embeddings.embedDocuments(finalDocs.map(d => d.pageContent));
        console.log(`Generated ${vectors.length} embeddings`);

        // Save embeddings with metadata
        const embeddingsWithMetadata = finalDocs.map((doc, index) => ({
            id: `chunk-${index}`,
            text: doc.pageContent,
            embedding: vectors[index],
            metadata: doc.metadata,
        }));

        fs.writeFileSync(outputPath, JSON.stringify(embeddingsWithMetadata, null, 2), "utf-8");
        console.log(`Saved ${embeddingsWithMetadata.length} embeddings to ${outputPath}`);
    } catch (error) {
        console.error("Error during PDF ingestion:", error);
        throw error;
    }
}

// $ Load embeddings from file
async function loadEmbeddings(filePath: string): Promise<Array<{ id: string; embedding: number[]; text: string; metadata: any }>> {
    try {
        console.log(`Loading embeddings from ${filePath}...`);
        const rawData = fs.readFileSync(filePath, "utf-8");
        const embeddingsData = JSON.parse(rawData);

        const items = embeddingsData.map((item: any) => ({
            id: item.id,
            embedding: (item.embedding || []).map((v: any) => Number(v)),
            text: item.text,
            metadata: typeof item.metadata === 'string' ? JSON.parse(item.metadata) : item.metadata,
        }));

        console.log(`Loaded ${items.length} embeddings from file.`);
        return items;
    } catch (error) {
        console.error("Error loading embeddings:", error);
        throw error;
    }
}

// $ Query the embeddings and get response from LLM
async function queryEmbeddings(question: string, embeddingsPath: string): Promise<void> {
    try {
        // Load embeddings
        const items = await loadEmbeddings(embeddingsPath);

        // Generate query embedding
        console.log("Computing embedding for the query...");
        const embeddingsClient = new OllamaEmbeddings({
            model: DEFAULT_EMBEDDING_MODEL,
            baseUrl: DEFAULT_OLLAMA_URL,
        });

        const qEmbedding = await embeddingsClient.embedQuery(question);

        if (!Array.isArray(qEmbedding) || qEmbedding.length === 0) {
            throw new Error("Failed to compute query embedding");
        }

        // Compute similarities
        const validItems = items.filter((it) => Array.isArray(it.embedding) && it.embedding.length > 0);
        const skipped = items.length - validItems.length;
        if (skipped > 0) console.warn(`Skipped ${skipped} items with missing/invalid embeddings.`);

        const sims = validItems.map((it) => ({
            id: it.id,
            score: cosine(qEmbedding, it.embedding),
            text: it.text,
            metadata: it.metadata
        }));

        sims.sort((a, b) => b.score - a.score);

        const TOP_K = Number(process.env.TOP_K || DEFAULT_TOP_K);
        const top = sims.slice(0, TOP_K);

        console.log(`Top ${top.length} chunks:`);
        top.forEach((t, i) => {
            // Handle both string and object metadata
            let pageNumber = "Unknown";
            if (typeof t.metadata === 'string') {
                try {
                    const parsedMetadata = JSON.parse(t.metadata);
                    pageNumber = parsedMetadata?.loc?.pageNumber || parsedMetadata?.page || "Unknown";
                } catch (e) {
                    // If parsing fails, use "Unknown"
                    pageNumber = "Unknown";
                }
            } else {
                pageNumber = t.metadata?.loc?.pageNumber || t.metadata?.page || "Unknown";
            }
            console.log(`${i + 1}. id=${t.id} score=${t.score.toFixed(4)} page=${pageNumber}`);
        });

        // Build context
        const context = top.map((t, i) =>
            `---chunk ${i + 1} (id=${t.id}, score=${t.score.toFixed(4)})---\n${t.text}`
        ).join("\n\n");

        // Get response from LLM
        console.log("Generating response from LLM...");
        const prompt = `You are a helpful assistant. Use the following context to answer the user's question thoroughly and in detail. If the answer is not contained in the context, say you don't know.

CONTEXT:
${context}

QUESTION:
${question}

`;

        const llm = new ChatOllama({ model: DEFAULT_CHAT_MODEL, temperature: 0.1 });
        const aiMsg = await llm.invoke(prompt);
        const response = aiMsg?.content ?? JSON.stringify(aiMsg);

        console.log("\n---Assistant response---\n");
        console.log(response);
    } catch (error) {
        console.error("Error during query processing:", error);
        throw error;
    }
}

// $ Interactive CLI
async function interactiveCLI(): Promise<void> {
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (query: string) => new Promise<string>((resolve) => readline.question(query, resolve));

    console.log("=== RAG System ===");
    console.log("Choose an option:");
    console.log("1. Ingest PDF and create embeddings");
    console.log("2. Query existing embeddings");
    console.log("3. Exit");

    const choice = await question("Enter your choice (1-3): ");

    switch (choice.trim()) {
        case "1":
            const pdfPath = await question("Enter PDF file path: ");
            const outputPath = await question("Enter output JSON file path (default: ./embeddings.json): ") || "./embeddings.json";

            try {
                await ingestPDF(pdfPath, outputPath);
            } catch (error) {
                console.error("Failed to ingest PDF:", error);
            }
            break;

        case "2":
            const embeddingsPath = await question("Enter embeddings JSON file path (default: ./embeddings.json): ") || "./embeddings.json";
            const query = await question("Enter your question: ");

            try {
                await queryEmbeddings(query, embeddingsPath);
            } catch (error) {
                console.error("Failed to process query:", error);
            }
            break;

        case "3":
            console.log("Goodbye!");
            readline.close();
            return;

        default:
            console.log("Invalid choice. Please try again.");
    }

    readline.close();
}

// Main function
async function main(): Promise<void> {
    // If command line arguments are provided, use them
    const args = process.argv.slice(2);

    if (args.length === 0) {
        // No arguments, run interactive CLI
        await interactiveCLI();
        return;
    }

    // Parse command line arguments
    if (args[0] === "--ingest" && args.length >= 3) {
        // Ingest mode: --ingest <pdf-path> <output-path>
        const pdfPath = args[1];
        const outputPath = args[2];
        await ingestPDF(pdfPath, outputPath);
    } else if (args[0] === "--query" && args.length >= 2) {
        // Query mode: --query <question> [embeddings-path]
        const question = args[1];
        const embeddingsPath = args[2] || "./embeddings.json";
        await queryEmbeddings(question, embeddingsPath);
    } else {
        console.log("Usage:");
        console.log("  bun run src/lib/rag.ts --ingest <pdf-path> <output-path>");
        console.log("  bun run src/lib/rag.ts --query <question> [embeddings-path]");
        console.log("  bun run src/lib/rag.ts (interactive mode)");
        process.exit(1);
    }
}

// Run the main function
if (require.main === module) {
    main().catch((error) => {
        console.error("Application error:", error);
        process.exit(1);
    });
}

// Export functions for potential use in other modules
export { ingestPDF, loadEmbeddings, queryEmbeddings, cosine };

// $ PDF's with Images should be removed
// ! https://pdf.imagestool.com/remove-image