import "dotenv/config";

import { mkdirSync, writeFileSync } from "fs";
import { resolve } from "path";

import { deployToAkash, type DeployResult } from "./akash_deploy";

type MatrixProfile = {
    profile: string;
    trainEntrypoint: string;
};

type ProfileManifestRecord = {
    profile: string;
    status: "success" | "failed";
    dseq: string | null;
    provider: string | null;
    akt_hour_estimate: number | null;
    cap_akt_hour: number | null;
    service_uri: string | null;
    error: string | null;
};

const PROFILES: MatrixProfile[] = [
    {
        profile: "rapid_a",
        trainEntrypoint:
            "python3 -m backend.scripts.local_training.train --data-root /app/AutoHDR/data --output-dir /app/AutoHDR/checkpoints --model micro_unet --epochs 8 --batch-size 8 --img-size 256 --lr 0.001 --max-train 10000 --max-val 1000 --num-workers 4",
    },
    {
        profile: "rapid_b",
        trainEntrypoint:
            "python3 -m backend.scripts.local_training.train --data-root /app/AutoHDR/data --output-dir /app/AutoHDR/checkpoints --model micro_unet --epochs 8 --batch-size 6 --img-size 320 --lr 0.001 --max-train 10000 --max-val 1000 --num-workers 4",
    },
    {
        profile: "rapid_c",
        trainEntrypoint:
            "python3 -m backend.scripts.local_training.train --data-root /app/AutoHDR/data --output-dir /app/AutoHDR/checkpoints --model micro_unet --epochs 10 --batch-size 8 --img-size 256 --lr 0.0005 --max-train 12000 --max-val 1200 --num-workers 4",
    },
];

function parsePositiveFloat(name: string, raw: string): number {
    const value = Number.parseFloat(raw);
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`Invalid ${name} value '${raw}'`);
    }
    return value;
}

function parsePositiveInt(name: string, raw: string): number {
    const value = Number.parseInt(raw, 10);
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`Invalid ${name} value '${raw}'`);
    }
    return value;
}

function nowSlug(): string {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hour = String(date.getHours()).padStart(2, "0");
    const minute = String(date.getMinutes()).padStart(2, "0");
    const second = String(date.getSeconds()).padStart(2, "0");
    return `${year}${month}${day}_${hour}${minute}${second}`;
}

async function launchProfile(
    profile: MatrixProfile,
    perJobCapAktHour: number
): Promise<ProfileManifestRecord> {
    try {
        const result: DeployResult = await deployToAkash({
            profileName: profile.profile,
            trainEntrypoint: profile.trainEntrypoint,
            maxAktPerHour: perJobCapAktHour,
            maxUaktPerBlock: Math.max(1, Math.floor((perJobCapAktHour * 1_000_000) / 600)),
        });

        return {
            profile: profile.profile,
            status: "success",
            dseq: result.dseq,
            provider: result.provider,
            akt_hour_estimate: result.akt_hour_estimate,
            cap_akt_hour: result.cap_akt_hour,
            service_uri: result.service_uri,
            error: null,
        };
    } catch (error) {
        return {
            profile: profile.profile,
            status: "failed",
            dseq: null,
            provider: null,
            akt_hour_estimate: null,
            cap_akt_hour: perJobCapAktHour,
            service_uri: null,
            error: error instanceof Error ? error.message : String(error),
        };
    }
}

async function main(): Promise<void> {
    const totalCapRaw = (process.env.AKASH_MAX_AKT_PER_HOUR_TOTAL || "4.0").trim();
    const deploymentCountRaw = (process.env.AKASH_DEPLOYMENT_COUNT || "3").trim();
    const totalCapAktHour = parsePositiveFloat("AKASH_MAX_AKT_PER_HOUR_TOTAL", totalCapRaw);
    const deploymentCount = parsePositiveInt("AKASH_DEPLOYMENT_COUNT", deploymentCountRaw);

    if (deploymentCount !== PROFILES.length) {
        throw new Error(
            `AKASH_DEPLOYMENT_COUNT=${deploymentCount} must match profile count (${PROFILES.length}) for matrix launch`
        );
    }

    const perJobCapAktHour = totalCapAktHour / deploymentCount;
    const startedAt = new Date().toISOString();

    console.log("=== Akash Matrix Launch ===");
    console.log(`Profiles: ${PROFILES.map((p) => p.profile).join(", ")}`);
    console.log(`Total cap: ${totalCapAktHour.toFixed(3)} AKT/hr`);
    console.log(`Per-job cap: ${perJobCapAktHour.toFixed(6)} AKT/hr`);

    const records = await Promise.all(PROFILES.map((profile) => launchProfile(profile, perJobCapAktHour)));

    const successCount = records.filter((r) => r.status === "success").length;
    const failedCount = records.length - successCount;
    const manifest = {
        timestamp_utc: new Date().toISOString(),
        started_at_utc: startedAt,
        total_cap_akt_hour: Number(totalCapAktHour.toFixed(6)),
        deployment_count: deploymentCount,
        per_job_cap_akt_hour: Number(perJobCapAktHour.toFixed(6)),
        profiles: records,
        success_count: successCount,
        failed_count: failedCount,
    };

    mkdirSync(resolve(process.cwd(), "logs"), { recursive: true });
    const manifestPath = resolve(process.cwd(), "logs", `akash_matrix_${nowSlug()}.json`);
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

    console.log(`Manifest: ${manifestPath}`);
    console.log(JSON.stringify(manifest, null, 2));

    if (successCount < PROFILES.length) {
        process.exit(1);
    }
}

void main().catch((error) => {
    console.error("❌ Matrix launch failed:", error instanceof Error ? error.message : String(error));
    process.exit(1);
});
