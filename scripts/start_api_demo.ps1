param(
    [string]$TokenFile = (Join-Path $env:TEMP "pcb-api-token.txt"),
    [int]$Port = 8000
)

$env:PCB_API_TOKEN = Get-Content -Raw -LiteralPath $TokenFile
$env:LANGCHAIN_TRACING_V2 = "false"
& (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe") -m pcb_agent.api --host 127.0.0.1 --port $Port
