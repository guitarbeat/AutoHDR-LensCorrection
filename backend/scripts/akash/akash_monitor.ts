import "dotenv/config";

const API_BASE_URL = "https://console-api.akash.network";

type MonitorArgs = {
    dseq: string;
    pollSeconds: number;
    maxIterations: number | null;
    showLogs: boolean;
    tailLines: number;
    costCapAktHour: number;
};

function printUsage(): void {
    console.log(
        [
            "Akash deployment monitor",
            "",
            "Usage:",
            "  npm run monitor -- --dseq <deployment-sequence> [options]",
            "",
            "Options:",
            "  --poll-seconds <n>   Poll interval in seconds (default: 30)",
            "  --iterations <n>     Stop after n polls (default: run forever)",
            "  --no-logs            Skip log tail fetching",
            "  --tail-lines <n>     Maximum new log lines to print per poll (default: 40)",
            "  --cost-cap <n>       Alert when estimated AKT/hr exceeds this value (default: 0.5)",
            "  -h, --help           Show this help",
        ].join("\n")
    );
}

function parseIntArg(value: string, flag: string): number {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid value for ${flag}: ${value}`);
    }
    return parsed;
}

function parseFloatArg(value: string, flag: string): number {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid value for ${flag}: ${value}`);
    }
    return parsed;
}

function parseArgs(argv: string[]): MonitorArgs {
    let dseq = "";
    let pollSeconds = 30;
    let maxIterations: number | null = null;
    let showLogs = true;
    let tailLines = 40;
    let costCapAktHour = 0.5;

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "-h" || arg === "--help") {
            printUsage();
            process.exit(0);
        }
        if (arg === "--no-logs") {
            showLogs = false;
            continue;
        }
        if (arg === "--dseq") {
            dseq = argv[i + 1] || "";
            i++;
            continue;
        }
        if (arg === "--poll-seconds") {
            pollSeconds = parseIntArg(argv[i + 1] || "", arg);
            i++;
            continue;
        }
        if (arg === "--iterations") {
            maxIterations = parseIntArg(argv[i + 1] || "", arg);
            i++;
            continue;
        }
        if (arg === "--tail-lines") {
            tailLines = parseIntArg(argv[i + 1] || "", arg);
            i++;
            continue;
        }
        if (arg === "--cost-cap") {
            costCapAktHour = parseFloatArg(argv[i + 1] || "", arg);
            i++;
            continue;
        }
        throw new Error(`Unknown argument: ${arg}`);
    }

    if (!dseq) {
        throw new Error("Missing required flag: --dseq");
    }

    return {
        dseq,
        pollSeconds,
        maxIterations,
        showLogs,
        tailLines,
        costCapAktHour,
    };
}

function getPathValue(obj: unknown, path: string): unknown {
    if (!obj || typeof obj !== "object") {
        return undefined;
    }
    const parts = path.split(".");
    let current: any = obj;
    for (const part of parts) {
        if (current && typeof current === "object" && part in current) {
            current = current[part];
        } else {
            return undefined;
        }
    }
    return current;
}

function extractState(obj: unknown): string {
    const candidates = [
        "state",
        "status",
        "data.state",
        "data.status",
        "deployment.state",
        "deployment.status",
        "data.deployment.state",
        "data.deployment.status",
    ];
    for (const path of candidates) {
        const value = getPathValue(obj, path);
        if (typeof value === "string" && value.trim()) {
            return value;
        }
    }
    return "unknown";
}

function extractProvider(obj: unknown): string | null {
    const candidates = [
        "provider",
        "data.provider",
        "data.leases.0.id.provider",
        "leases.0.id.provider",
        "deployment.provider",
        "deployment.id.provider",
        "data.deployment.provider",
        "data.deployment.id.provider",
    ];
    for (const path of candidates) {
        const value = getPathValue(obj, path);
        if (typeof value === "string" && value.trim()) {
            return value;
        }
    }
    return null;
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

class ApiRequestError extends Error {
    status: number;
    endpoint: string;
    body: string;

    constructor(status: number, endpoint: string, body: string) {
        super(`API error ${status} on ${endpoint}: ${body}`);
        this.status = status;
        this.endpoint = endpoint;
        this.body = body;
    }
}

async function apiRequest<T>(endpoint: string): Promise<T> {
    const apiKey = process.env.AKASH_API_KEY;
    if (!apiKey) {
        throw new Error("Missing AKASH_API_KEY in .env");
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
        },
    });

    if (!response.ok) {
        const body = await response.text();
        throw new ApiRequestError(response.status, endpoint, body);
    }

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return (await response.json()) as T;
    }
    return (await response.text()) as unknown as T;
}

type BidData = {
    bid?: {
        id?: {
            provider?: string;
        };
        price?: {
            amount?: string;
            denom?: string;
        };
    };
};

function hourlyAktFromBidAmount(amount: string | undefined): number | null {
    if (!amount) {
        return null;
    }
    const amountPerBlock = Number.parseInt(amount, 10);
    if (!Number.isFinite(amountPerBlock)) {
        return null;
    }
    return (amountPerBlock * 600) / 1_000_000;
}

async function fetchBids(dseq: string): Promise<BidData[]> {
    const response = await apiRequest<{ data?: BidData[] }>(`/v1/bids?dseq=${encodeURIComponent(dseq)}`);
    return response.data || [];
}

async function fetchDeployment(dseq: string): Promise<unknown> {
    const endpoints = [`/v1/deployments/${encodeURIComponent(dseq)}`, `/v1/deployments?dseq=${encodeURIComponent(dseq)}`];
    let lastError: Error | null = null;
    for (const endpoint of endpoints) {
        try {
            return await apiRequest<unknown>(endpoint);
        } catch (error) {
            lastError = error as Error;
        }
    }
    throw lastError || new Error("Unable to fetch deployment details");
}

function normalizeLogLines(payload: unknown): string[] {
    if (typeof payload === "string") {
        return payload.split(/\r?\n/);
    }
    if (!payload || typeof payload !== "object") {
        return [];
    }

    const candidates = [
        getPathValue(payload, "logs"),
        getPathValue(payload, "data"),
        getPathValue(payload, "data.logs"),
    ];

    for (const candidate of candidates) {
        if (typeof candidate === "string") {
            return candidate.split(/\r?\n/);
        }
        if (Array.isArray(candidate)) {
            return candidate.map((line) => String(line));
        }
    }

    return JSON.stringify(payload, null, 2).split(/\r?\n/);
}

type LogResult = {
    lines: string[];
    available: boolean;
    endpoint?: string;
};

async function fetchLogs(dseq: string, provider?: string | null): Promise<LogResult> {
    const encodedDseq = encodeURIComponent(dseq);
    const encodedProvider = provider ? encodeURIComponent(provider) : "";
    const endpoints = [
        `/v1/deployments/${encodedDseq}/logs`,
        provider ? `/v1/deployments/${encodedDseq}/logs?provider=${encodedProvider}` : "",
        `/v1/leases/${encodedDseq}/logs`,
    ].filter(Boolean);

    let lastError: Error | null = null;
    for (const endpoint of endpoints) {
        try {
            const raw = await apiRequest<unknown>(endpoint);
            return {
                lines: normalizeLogLines(raw).filter((line) => line.trim().length > 0),
                available: true,
                endpoint,
            };
        } catch (error) {
            if (error instanceof ApiRequestError && error.status === 404) {
                lastError = error;
                continue;
            }
            throw error;
        }
    }

    if (lastError) {
        return { lines: [], available: false };
    }
    return { lines: [], available: false };
}

async function main(): Promise<void> {
    let args: MonitorArgs;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (error) {
        console.error(`Argument error: ${error instanceof Error ? error.message : String(error)}`);
        printUsage();
        process.exit(1);
        return;
    }

    console.log("Monitoring Akash deployment");
    console.log(`DSEQ: ${args.dseq}`);
    console.log(`Poll interval: ${args.pollSeconds}s`);
    console.log(`Log streaming: ${args.showLogs ? "enabled" : "disabled"}`);
    if (args.maxIterations !== null) {
        console.log(`Max iterations: ${args.maxIterations}`);
    }
    console.log(`Cost alert cap: ${args.costCapAktHour.toFixed(3)} AKT/hr`);
    console.log("");

    let iteration = 0;
    let seenLogCount = 0;

    while (args.maxIterations === null || iteration < args.maxIterations) {
        iteration += 1;
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] Poll ${iteration}`);
        let resolvedProvider: string | null = null;

        try {
            const [deployment, bids] = await Promise.all([fetchDeployment(args.dseq), fetchBids(args.dseq)]);
            const deploymentState = extractState(deployment);
            const provider = extractProvider(deployment);

            const bestBid = bids[0];
            const bidProvider = bestBid?.bid?.id?.provider || null;
            const bidAmount = bestBid?.bid?.price?.amount;
            const hourlyAkt = hourlyAktFromBidAmount(bidAmount);
            resolvedProvider = provider || bidProvider;

            console.log(`State: ${deploymentState}`);
            if (resolvedProvider) {
                console.log(`Provider: ${resolvedProvider}`);
            }
            if (hourlyAkt !== null) {
                console.log(`Estimated cost: ${hourlyAkt.toFixed(3)} AKT/hr`);
                if (hourlyAkt > args.costCapAktHour) {
                    console.log(
                        `ALERT: estimated cost exceeds cap (${hourlyAkt.toFixed(3)} > ${args.costCapAktHour.toFixed(3)} AKT/hr)`
                    );
                }
            } else if (bidAmount) {
                console.log(`Bid amount (raw): ${bidAmount} uakt/block`);
            }
        } catch (error) {
            console.log(`Status check failed: ${error instanceof Error ? error.message : String(error)}`);
        }

        if (args.showLogs) {
            try {
                const logs = await fetchLogs(args.dseq, resolvedProvider);
                if (!logs.available) {
                    console.log("Logs: endpoint not available yet (provider may still be initializing)");
                    continue;
                }

                const lines = logs.lines;
                if (lines.length < seenLogCount) {
                    seenLogCount = 0;
                }
                const newLines = lines.slice(seenLogCount);
                seenLogCount = lines.length;

                if (newLines.length === 0) {
                    console.log(`Logs: no new lines (${logs.endpoint})`);
                } else {
                    const tail = newLines.slice(-args.tailLines);
                    console.log(`Logs: ${newLines.length} new lines from ${logs.endpoint} (showing ${tail.length})`);
                    for (const line of tail) {
                        console.log(`  ${line}`);
                    }
                }
            } catch (error) {
                console.log(`Log fetch failed: ${error instanceof Error ? error.message : String(error)}`);
            }
        }

        if (args.maxIterations === null || iteration < args.maxIterations) {
            console.log("");
            await sleep(args.pollSeconds * 1000);
        }
    }

    console.log("Monitoring complete.");
}

void main();
