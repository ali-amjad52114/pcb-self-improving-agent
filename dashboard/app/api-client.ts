export type ApiRun = Record<string, unknown>;

const configuredBase = process.env.NEXT_PUBLIC_PCB_API_BASE_URL?.trim();
// Same-origin proxy is the secure default. It owns the backend secret and CORS.
export const apiBase = (configuredBase || "/api/backend").replace(/\/$/, "");

async function request(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) throw new Error(`PCB API ${response.status}`);
  return response.json();
}

export async function apiHealthy(): Promise<boolean> {
  try {
    const value = await request("/health");
    const health = value as Record<string, unknown>;
    return health.ok !== false && health.status !== "unhealthy";
  } catch {
    return false;
  }
}

export async function listApiRuns(): Promise<ApiRun[]> {
  const value = await request("/runs") as ApiRun[] | { runs?: ApiRun[] };
  return Array.isArray(value) ? value : value.runs ?? [];
}

export async function startApiRun(mode: "memory" | "cold"): Promise<ApiRun> {
  const value = await request("/runs", {
    method: "POST",
    body: JSON.stringify({ mode }),
  }) as ApiRun | { run?: ApiRun };
  return ("run" in value && value.run ? value.run : value) as ApiRun;
}

export async function getApiRun(id: string): Promise<ApiRun> {
  const value = await request(`/runs/${encodeURIComponent(id)}`) as ApiRun | { run?: ApiRun };
  return ("run" in value && value.run ? value.run : value) as ApiRun;
}
