import "dotenv/config";

const API_BASE_URL = "https://console-api.akash.network";

type KaggleAuthMode = "token" | "keypair";
type TrainingMode = "remote_script" | "repo_fallback";

export type DeployConfig = {
    apiKey: string;
    profileName: string;
    depositAmount: number;
    maxAktPerHour: number;
    maxUaktPerBlock: number;
    kaggleAuthMode: KaggleAuthMode;
    kaggleToken?: string;
    kaggleUsername?: string;
    kaggleKey?: string;
    trainScriptUrl?: string;
    trainRepoUrl: string;
    trainEntrypoint: string;
    trainingMode: TrainingMode;
    providerDenylist: Set<string>;
    kaggleDownloadMaxAttempts: number;
    kaggleDownloadRetrySeconds: number;
};

export type DeployResult = {
    status: "success";
    profile: string;
    dseq: string;
    provider: string;
    price_amount: string;
    price_denom: string;
    akt_hour_estimate: number;
    cap_akt_hour: number;
    max_uakt_per_block: number;
    service_uri: string | null;
    training_mode: TrainingMode;
};

type RawBid = {
    bid?: {
        id?: {
            provider?: string;
            gseq?: number;
            oseq?: number;
        };
        price?: {
            amount?: string;
            denom?: string;
        };
    };
};

type DeploymentCreateResponse = {
    data: {
        dseq: string;
        manifest: string;
    };
};

function parsePositiveFloat(name: string, raw: string): number {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid ${name} value '${raw}'`);
    }
    return parsed;
}

function parsePositiveInt(name: string, raw: string): number {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid ${name} value '${raw}'`);
    }
    return parsed;
}

function shellSingleQuote(value: string): string {
    return `'${value.replace(/'/g, `'\"'\"'`)}'`;
}

function parseOptionalInt(name: string, raw: string | undefined, fallback: number): number {
    if (!raw || !raw.trim()) return fallback;
    const parsed = Number.parseInt(raw.trim(), 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`Invalid ${name} value '${raw}'`);
    }
    return parsed;
}

function parseProviderDenylist(raw: string): Set<string> {
    return new Set(
        raw
            .split(",")
            .map((item) => item.trim())
            .filter((item) => item.length > 0)
    );
}

function derivePerJobCapAktPerHour(env: NodeJS.ProcessEnv): number {
    const explicit = (env.AKASH_MAX_AKT_PER_HOUR || "").trim();
    if (explicit) {
        return parsePositiveFloat("AKASH_MAX_AKT_PER_HOUR", explicit);
    }

    const totalRaw = (env.AKASH_MAX_AKT_PER_HOUR_TOTAL || "4.0").trim();
    const countRaw = (env.AKASH_DEPLOYMENT_COUNT || "3").trim();
    const total = parsePositiveFloat("AKASH_MAX_AKT_PER_HOUR_TOTAL", totalRaw);
    const count = parsePositiveInt("AKASH_DEPLOYMENT_COUNT", countRaw);
    return total / count;
}

export function resolveDeployConfig(
    overrides: Partial<DeployConfig> = {},
    env: NodeJS.ProcessEnv = process.env
): DeployConfig {
    const apiKey = overrides.apiKey ?? (env.AKASH_API_KEY || "").trim();
    if (!apiKey) {
        throw new Error("Missing AKASH_API_KEY in .env");
    }

    const kaggleToken = overrides.kaggleToken ?? (env.KAGGLE_API_TOKEN || "").trim();
    const kaggleUsername = overrides.kaggleUsername ?? (env.KAGGLE_USERNAME || "").trim();
    const kaggleKey = overrides.kaggleKey ?? (env.KAGGLE_KEY || "").trim();

    let kaggleAuthMode: KaggleAuthMode;
    if (kaggleToken) {
        kaggleAuthMode = "token";
    } else if (kaggleUsername && kaggleKey) {
        kaggleAuthMode = "keypair";
    } else {
        throw new Error(
            "Missing Kaggle auth. Set KAGGLE_API_TOKEN (preferred) or KAGGLE_USERNAME + KAGGLE_KEY."
        );
    }

    const trainScriptUrl = overrides.trainScriptUrl ?? (env.AKASH_TRAIN_SCRIPT_URL || "").trim();
    const trainRepoUrl =
        overrides.trainRepoUrl ??
        (env.AKASH_TRAIN_REPO_URL || "https://github.com/guitarbeat/AutoHDR-LensCorrection.git").trim();
    const trainEntrypoint =
        overrides.trainEntrypoint ??
        (
            env.AKASH_TRAIN_ENTRYPOINT ||
            "python3 -m backend.scripts.local_training.train --data-root /app/AutoHDR/data --output-dir /app/AutoHDR/checkpoints --model micro_unet --epochs 8 --batch-size 8 --img-size 256 --lr 0.001 --max-train 10000 --max-val 1000 --num-workers 4"
        ).trim();

    if (!trainScriptUrl && !trainRepoUrl) {
        throw new Error("Missing training source. Set AKASH_TRAIN_SCRIPT_URL or AKASH_TRAIN_REPO_URL.");
    }

    const maxAktPerHour = overrides.maxAktPerHour ?? derivePerJobCapAktPerHour(env);
    const maxUaktPerBlock = overrides.maxUaktPerBlock ?? Math.max(1, Math.floor((maxAktPerHour * 1_000_000) / 600));

    const profileName = overrides.profileName ?? (env.AKASH_PROFILE_NAME || "default");
    const depositAmount =
        overrides.depositAmount ?? parsePositiveFloat("AKASH_DEPOSIT_AMOUNT", (env.AKASH_DEPOSIT_AMOUNT || "5").trim());

    const trainingMode: TrainingMode = trainScriptUrl ? "remote_script" : "repo_fallback";
    const providerDenylist = overrides.providerDenylist ?? parseProviderDenylist(env.AKASH_PROVIDER_DENYLIST || "");
    const kaggleDownloadMaxAttempts =
        overrides.kaggleDownloadMaxAttempts ??
        parseOptionalInt(
            "AKASH_KAGGLE_DOWNLOAD_MAX_ATTEMPTS",
            env.AKASH_KAGGLE_DOWNLOAD_MAX_ATTEMPTS,
            30
        );
    if (!Number.isFinite(kaggleDownloadMaxAttempts) || kaggleDownloadMaxAttempts <= 0) {
        throw new Error(`Invalid kaggleDownloadMaxAttempts value '${kaggleDownloadMaxAttempts}'`);
    }
    const kaggleDownloadRetrySeconds =
        overrides.kaggleDownloadRetrySeconds ??
        parseOptionalInt(
            "AKASH_KAGGLE_DOWNLOAD_RETRY_SECONDS",
            env.AKASH_KAGGLE_DOWNLOAD_RETRY_SECONDS,
            60
        );
    if (!Number.isFinite(kaggleDownloadRetrySeconds) || kaggleDownloadRetrySeconds <= 0) {
        throw new Error(`Invalid kaggleDownloadRetrySeconds value '${kaggleDownloadRetrySeconds}'`);
    }
    return {
        apiKey,
        profileName,
        depositAmount,
        maxAktPerHour,
        maxUaktPerBlock,
        kaggleAuthMode,
        kaggleToken,
        kaggleUsername,
        kaggleKey,
        trainScriptUrl: trainScriptUrl || undefined,
        trainRepoUrl,
        trainEntrypoint,
        trainingMode,
        providerDenylist,
        kaggleDownloadMaxAttempts,
        kaggleDownloadRetrySeconds,
    };
}

function buildKaggleAuthCommand(config: DeployConfig): string {
    if (config.kaggleAuthMode === "token") {
        return [
            "mkdir -p ~/.kaggle ~/.config/kaggle",
            `export KAGGLE_API_TOKEN=${shellSingleQuote(config.kaggleToken || "")}`,
            "export KAGGLE_CONFIG_DIR=~/.config/kaggle",
        ].join(" && ");
    }
    const credsJson = JSON.stringify({
        username: config.kaggleUsername || "",
        key: config.kaggleKey || "",
    });
    return [
        "mkdir -p ~/.kaggle ~/.config/kaggle",
        `printf %s ${shellSingleQuote(credsJson)} > ~/.kaggle/kaggle.json`,
        `printf %s ${shellSingleQuote(credsJson)} > ~/.config/kaggle/kaggle.json`,
        "chmod 600 ~/.kaggle/kaggle.json",
        "chmod 600 ~/.config/kaggle/kaggle.json",
        "export KAGGLE_CONFIG_DIR=~/.config/kaggle",
    ].join(" && ");
}

function sanitizeRunName(name: string): string {
    const cleaned = name.replace(/[^a-zA-Z0-9_-]/g, "_");
    return cleaned.length > 0 ? cleaned : "run";
}

function buildRepoTrainingRuntimeCommand(config: DeployConfig): string {
    const runName = sanitizeRunName(config.profileName) + "_inference";
    const bestCheckpointPath = "/app/AutoHDR/checkpoints/best_model.pt";
    const metricsJsonPath = "/app/AutoHDR/artifacts/evaluation/best_model_metrics.json";
    const inferenceZipPath = `/app/AutoHDR/artifacts/inference/${runName}/${runName}.zip`;
    const inferenceSummaryPath = `/app/AutoHDR/artifacts/inference/${runName}/summary.json`;
    const manifestTxtPath = `/app/AutoHDR/serve/${runName}_artifacts.txt`;

    return [
        `echo "Cloning training repo: ${config.trainRepoUrl}"`,
        `git clone --depth 1 ${shellSingleQuote(config.trainRepoUrl)} /app/AutoHDR/repo`,
        "cd /app/AutoHDR/repo",
        "pip install -r requirements.txt",
        "mkdir -p /app/AutoHDR/checkpoints /app/AutoHDR/artifacts /app/AutoHDR/serve",
        "export AUTOHDR_DATA_ROOT=/app/AutoHDR/data",
        "export AUTOHDR_CHECKPOINT_ROOT=/app/AutoHDR/checkpoints",
        "export AUTOHDR_OUTPUT_ROOT=/app/AutoHDR/artifacts",
        `echo "Starting training profile ${config.profileName}"`,
        config.trainEntrypoint,
        `test -f ${bestCheckpointPath}`,
        `python3 -m backend.scripts.local_training.evaluate_model --checkpoint ${bestCheckpointPath} --data-root /app/AutoHDR/data --output-root /app/AutoHDR/artifacts --batch-size 4 --num-workers 2 --max-val 1000 --device cuda`,
        `python3 -m backend.scripts.local_training.inference --checkpoint ${bestCheckpointPath} --test-dir /app/AutoHDR/data/test-originals --output-root /app/AutoHDR/artifacts --run-name ${runName} --device cuda`,
        `printf '%s\\n' 'profile=${config.profileName}' 'checkpoint=${bestCheckpointPath}' 'metrics_json=${metricsJsonPath}' 'inference_zip=${inferenceZipPath}' 'inference_summary=${inferenceSummaryPath}' > ${manifestTxtPath}`,
        "cp -r /app/AutoHDR/checkpoints /app/AutoHDR/serve/checkpoints",
        "cp -r /app/AutoHDR/artifacts /app/AutoHDR/serve/artifacts",
        "cd /app/AutoHDR/serve",
        "python3 -m http.server 8080",
    ].join(" && ");
}

function buildRemoteTrainingRuntimeCommand(config: DeployConfig): string {
    return [
        `echo "Downloading Python training script from ${config.trainScriptUrl}"`,
        `wget -qO kaggle_unet_train.py ${shellSingleQuote(config.trainScriptUrl || "")}`,
        "echo 'Starting remote script training'",
        "python3 kaggle_unet_train.py",
        "mkdir -p /app/AutoHDR/serve",
        "printf '%s\\n' 'remote_script_complete=1' > /app/AutoHDR/serve/remote_artifacts.txt",
        "cd /app/AutoHDR/serve",
        "python3 -m http.server 8080",
    ].join(" && ");
}

function buildTrainingRuntimeCommand(config: DeployConfig): string {
    return config.trainingMode === "remote_script"
        ? buildRemoteTrainingRuntimeCommand(config)
        : buildRepoTrainingRuntimeCommand(config);
}

function buildSDL(config: DeployConfig): string {
    const kaggleDownloadWithRetry = [
        `max_attempts=${config.kaggleDownloadMaxAttempts};`,
        "attempt=1;",
        "while true; do",
        "  kaggle competitions download -c automatic-lens-correction && break;",
        "  if [ \"$attempt\" -ge \"$max_attempts\" ]; then",
        "    echo \"Kaggle download failed after $attempt attempts\";",
        "    exit 1;",
        "  fi;",
        `  echo \"Kaggle download attempt $attempt failed; retrying in ${config.kaggleDownloadRetrySeconds} seconds...\";`,
        `  sleep ${config.kaggleDownloadRetrySeconds};`,
        "  attempt=$((attempt+1));",
        "done",
    ].join(" ");

    const bootstrapCommands = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update",
        "apt-get install -y git wget unzip jq curl",
        "rm -rf /var/lib/apt/lists/*",
        "pip install --no-cache-dir --ignore-requires-python kaggle==2.0.0",
        "pip install --no-cache-dir opencv-python-headless matplotlib seaborn tqdm pillow python-dotenv requests",
        "mkdir -p /app/AutoHDR/data /app/AutoHDR/serve",
        "cd /app/AutoHDR",
        buildKaggleAuthCommand(config),
        "echo 'Kaggle auth preflight...'",
        "kaggle competitions files -c automatic-lens-correction --page-size 1 >/tmp/kaggle_preflight.txt",
        "echo 'Kaggle preflight passed.'",
        "echo 'Downloading Kaggle dataset...'",
        kaggleDownloadWithRetry,
        "unzip -q automatic-lens-correction.zip -d data",
        "rm -f automatic-lens-correction.zip",
        "echo 'Dataset extracted. Launching training/runtime flow...'",
        buildTrainingRuntimeCommand(config),
    ];

    return `
version: "2.0"

services:
  miner:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    expose:
      - port: 8080
        as: 80
        to:
          - global: true
    command:
      - "bash"
      - "-c"
    args:
      - |
        ${bootstrapCommands.join(" &&\n        ")}

profiles:
  compute:
    miner:
      resources:
        cpu:
          units: 2.0
        memory:
          size: 16Gi
        storage:
          - size: 60Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
  placement:
    dcloud:
      pricing:
        miner:
          denom: uakt
          amount: ${config.maxUaktPerBlock}

deployment:
  miner:
    dcloud:
      profile: miner
      count: 1
`;
}

async function apiRequest<T>(apiKey: string, endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
    };
    if (options.headers) {
        Object.assign(headers, options.headers);
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers,
    });
    if (!response.ok) {
        const error = await response.text();
        throw new Error(`API error ${response.status} on ${endpoint}: ${error}`);
    }
    return response.json() as Promise<T>;
}

async function waitForBids(apiKey: string, dseq: string, maxAttempts = 20): Promise<RawBid[]> {
    for (let i = 0; i < maxAttempts; i++) {
        const response = await apiRequest<{ data?: RawBid[] }>(apiKey, `/v1/bids?dseq=${encodeURIComponent(dseq)}`);
        if (response.data && response.data.length > 0) {
            return response.data;
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
    }
    throw new Error("No bids received after maximum attempts");
}

function parseBidAmountPerBlock(bid: RawBid): number | null {
    const raw = bid.bid?.price?.amount;
    if (!raw) return null;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return null;
    }
    return parsed;
}

function pickAffordableBid(
    bids: RawBid[],
    maxAktPerHour: number,
    providerDenylist: Set<string>
): { bid: RawBid; aktPerHour: number; amountPerBlock: number } {
    const affordable = bids
        .map((bid) => {
            const amountPerBlock = parseBidAmountPerBlock(bid);
            if (amountPerBlock === null) return null;
            const aktPerHour = (amountPerBlock * 600) / 1_000_000;
            return { bid, aktPerHour, amountPerBlock };
        })
        .filter((item): item is { bid: RawBid; aktPerHour: number; amountPerBlock: number } => item !== null)
        .filter((item) => {
            const provider = item.bid.bid?.id?.provider || "";
            return !providerDenylist.has(provider);
        })
        .filter((item) => item.aktPerHour <= maxAktPerHour)
        .sort((a, b) => a.amountPerBlock - b.amountPerBlock);

    if (affordable.length === 0) {
        throw new Error(`No bids within cost cap (${maxAktPerHour.toFixed(3)} AKT/hr)`);
    }
    return affordable[0];
}

async function fetchDeploymentDetails(apiKey: string, dseq: string): Promise<unknown> {
    return apiRequest<unknown>(apiKey, `/v1/deployments/${encodeURIComponent(dseq)}`);
}

function extractServiceUri(payload: unknown): string | null {
    if (!payload || typeof payload !== "object") return null;
    const root = payload as any;
    const uri = root?.data?.leases?.[0]?.status?.services?.miner?.uris?.[0];
    return typeof uri === "string" && uri.trim() ? uri : null;
}

async function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function deployToAkash(overrides: Partial<DeployConfig> = {}): Promise<DeployResult> {
    const config = resolveDeployConfig(overrides);

    console.log(`\n=== Deploy profile: ${config.profileName} ===`);
    console.log(`Training mode: ${config.trainingMode}`);
    console.log(`Cost cap: ${config.maxAktPerHour.toFixed(3)} AKT/hr (pricing cap ${config.maxUaktPerBlock} uakt/block)`);
    console.log(`Kaggle auth mode: ${config.kaggleAuthMode}`);

    const sdl = buildSDL(config);
    const deployResponse = await apiRequest<DeploymentCreateResponse>(config.apiKey, "/v1/deployments", {
        method: "POST",
        body: JSON.stringify({
            data: {
                sdl,
                deposit: config.depositAmount,
            },
        }),
    });

    const dseq = deployResponse.data.dseq;
    const manifest = deployResponse.data.manifest;
    console.log(`Deployment created: dseq=${dseq}`);

    const bids = await waitForBids(config.apiKey, dseq);
    const picked = pickAffordableBid(bids, config.maxAktPerHour, config.providerDenylist);
    const bid = picked.bid;

    const provider = bid.bid?.id?.provider;
    const gseq = bid.bid?.id?.gseq;
    const oseq = bid.bid?.id?.oseq;
    const amount = bid.bid?.price?.amount;
    const denom = bid.bid?.price?.denom;

    if (!provider || gseq === undefined || oseq === undefined || !amount || !denom) {
        throw new Error("Selected bid is missing required lease fields");
    }

    console.log(`Selected provider: ${provider}`);
    console.log(`Estimated cost: ${picked.aktPerHour.toFixed(3)} AKT/hr`);

    await apiRequest(config.apiKey, "/v1/leases", {
        method: "POST",
        body: JSON.stringify({
            manifest,
            leases: [
                {
                    dseq,
                    gseq,
                    oseq,
                    provider,
                },
            ],
        }),
    });

    let serviceUri: string | null = null;
    for (let attempt = 1; attempt <= 6; attempt++) {
        try {
            const details = await fetchDeploymentDetails(config.apiKey, dseq);
            serviceUri = extractServiceUri(details);
            if (serviceUri) break;
        } catch (_error) {
            // Best-effort URI fetch; do not fail deployment success if this is temporarily unavailable.
        }
        await sleep(3000);
    }

    const result: DeployResult = {
        status: "success",
        profile: config.profileName,
        dseq,
        provider,
        price_amount: amount,
        price_denom: denom,
        akt_hour_estimate: Number(picked.aktPerHour.toFixed(6)),
        cap_akt_hour: Number(config.maxAktPerHour.toFixed(6)),
        max_uakt_per_block: config.maxUaktPerBlock,
        service_uri: serviceUri,
        training_mode: config.trainingMode,
    };

    console.log("Lease created successfully.");
    console.log(JSON.stringify(result, null, 2));
    return result;
}

async function main(): Promise<void> {
    try {
        await deployToAkash();
    } catch (error) {
        console.error("❌ Deployment failed:", error instanceof Error ? error.message : String(error));
        process.exit(1);
    }
}

if (require.main === module) {
    void main();
}
