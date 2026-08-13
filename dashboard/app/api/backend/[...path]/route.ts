const backendBase = () => {
  const value = process.env.BACKEND_BASE_URL?.trim();
  if (!value) throw new Error("Live backend is not configured");
  return value.replace(/\/$/, "");
};

async function proxy(request: Request, context: { params: Promise<{ path: string[] }> }) {
  try {
    const { path } = await context.params;
    const token = process.env.BACKEND_TOKEN?.trim();
    if (!token) throw new Error("Live backend token is not configured");
    const source = new URL(request.url);
    const target = `${backendBase()}/${path.join("/")}${source.search}`;
    const body = request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.arrayBuffer();
    const response = await fetch(target, {
      method: request.method,
      body,
      cache: "no-store",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": request.headers.get("content-type") || "application/json",
      },
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") || "application/json" },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend unavailable";
    return Response.json({ error: message }, { status: 503 });
  }
}

export const GET = proxy;
export const POST = proxy;
