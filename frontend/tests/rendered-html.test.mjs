import assert from "node:assert/strict";
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

test("server-renders the Switchback AI experience", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Switchback AI/i);
  assert.match(html, /SWITCHBACK AI/);
  assert.match(html, /DEMO FLOW/);
  assert.match(html, /MongoDB Atlas/);
  assert.match(html, /Every experiment makes the next one smarter/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("exposes honest demo and accessibility labels", async () => {
  const response = await render();
  const html = await response.text();

  assert.match(html, /GUIDED MOCK/);
  assert.match(html, /TRANSPARENT DEMO MODE/);
  assert.match(html, /aria-label="Project summary"/);
  assert.match(html, /aria-label="Technology evidence ledger"/);
});
