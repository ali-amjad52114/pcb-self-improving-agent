import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Traceboard observatory", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Traceboard/);
  assert.match(html, /PCB Autonomous Lab/);
  assert.match(html, /Run Cold/);
  assert.match(html, /Run With Memory/);
  assert.match(html, /Kaggle PCB defects/);
  assert.match(html, /2,953/);
  assert.match(html, /MongoDB|Atlas/);
  assert.match(html, /Fireworks/i);
  assert.doesNotMatch(html, /codex-preview|Building your site|react-loading-skeleton/i);
});

test("ships live API wiring with an explicit demo fallback", async () => {
  const [page, client, proxy, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api-client.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/backend/[...path]/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /Live API/);
  assert.match(page, /Run Cold/);
  assert.match(page, /Run With Memory/);
  assert.match(page, /Cold versus memory comparison/);
  assert.match(page, /launchDemoRun/);
  assert.match(page, /startApiRun/);
  assert.match(client, /\/api\/backend/);
  assert.match(client, /apiHealthy/);
  assert.match(client, /listApiRuns/);
  assert.match(client, /getApiRun/);
  assert.match(proxy, /BACKEND_BASE_URL/);
  assert.match(proxy, /BACKEND_TOKEN/);
  assert.match(proxy, /authorization:\s*`Bearer/);
  assert.match(layout, /Traceboard/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
