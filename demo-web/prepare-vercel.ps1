param(
    [Parameter(Mandatory)]
    [uri]$BackendOrigin
)

$ErrorActionPreference = 'Stop'
if ($BackendOrigin.Scheme -ne 'https' -or $BackendOrigin.AbsolutePath -ne '/' -or
    $BackendOrigin.Query -or $BackendOrigin.Fragment -or $BackendOrigin.UserInfo) {
    throw 'BackendOrigin must be an HTTPS origin, such as https://your-tunnel.trycloudflare.com.'
}

$origin = $BackendOrigin.GetLeftPart([System.UriPartial]::Authority)
$output = Join-Path $PSScriptRoot '.vercel-frontend'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'index.html') -Destination $output -Force

$routes = @('/api', '/timeline', '/preview', '/start', '/stop', '/frame',
            '/source/:id', '/clip/:id', '/saved/:id', '/discarded/:id')
$config = @{
    framework = $null
    rewrites = @($routes | ForEach-Object { @{ source = $_; destination = "$origin$_" } })
}
$json = $config | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $output 'vercel.json'), $json, [System.Text.UTF8Encoding]::new($false))
Write-Output $output
