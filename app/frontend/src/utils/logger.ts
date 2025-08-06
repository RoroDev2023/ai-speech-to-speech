// src/utils/logger.ts
export function logEvent(label: string, data: any) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${label}]`, data);
}
