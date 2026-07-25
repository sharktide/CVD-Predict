param(
    [Parameter(Mandatory=$true)]
    [string]$Url,

    [Parameter(Mandatory=$true)]
    [string]$OutFile
)

Add-Type -AssemblyName System.Net.Http

# Ensure directory exists
$dir = Split-Path $OutFile
if (-not (Test-Path $dir)) {
    throw "Directory does not exist: $dir"
}

# Check partial file
$existing = 0
if (Test-Path $OutFile) {
    $existing = (Get-Item $OutFile).Length
    Write-Host "Resuming from $existing bytes..."
}

# Create HttpClient
$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromDays(1)

# Create request
$request = New-Object System.Net.Http.HttpRequestMessage -ArgumentList @(
    [System.Net.Http.HttpMethod]::Get,
    $Url
)

# Add Range header if resuming
if ($existing -gt 0) {
    $request.Headers.Range = New-Object System.Net.Http.Headers.RangeHeaderValue($existing, $null)
}

$response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
$response.EnsureSuccessStatusCode()

$stream = $response.Content.ReadAsStreamAsync().Result

# SAFELY open file
try {
    $fs = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
} catch {
    throw "Failed to open output file: $OutFile. Error: $($_.Exception.Message)"
}

if ($fs -eq $null) {
    throw "File stream is null — cannot write to $OutFile"
}

# Download loop
$buffer = New-Object byte[] (1024 * 1024 * 4)   # 4 MB
$totalRead = $existing
$sessionRead = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()

while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
    $fs.Write($buffer, 0, $read)

    $totalRead += $read
    $sessionRead += $read

    $mb = [math]::Round($totalRead / 1MB, 2)
    $speed = [math]::Round(($sessionRead / $sw.Elapsed.TotalSeconds) / 1MB, 2)

    Write-Progress -Activity "Downloading..." -Status "$mb MB downloaded ($speed MB/s)"
}

$fs.Close()
$stream.Close()
