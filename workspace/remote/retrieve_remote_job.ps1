[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$HostName,
    [Parameter(Mandatory = $true)] [string]$SshUser,
    [Parameter(Mandatory = $true)] [string]$RemoteRoot,
    [Parameter(Mandatory = $true)] [ValidatePattern('^[A-Za-z0-9_.-]+$')] [string]$JobId
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
if ($projectRoot -notmatch '^[Hh]:\\') {
    throw "Refusing to retrieve outside the H: project tree: $projectRoot"
}

$localControlRoot = 'external_workspaces/step30_remote_control_v1'
$localTmp = Join-Path $localControlRoot 'tmp'
$localSsh = Join-Path $localControlRoot 'ssh'
New-Item -ItemType Directory -Force $localTmp, $localSsh | Out-Null
$env:TEMP = $localTmp
$env:TMP = $localTmp
$env:TMPDIR = $localTmp
$sshConfig = Join-Path $localSsh 'ssh_config_h_drive_v1'
if (-not (Test-Path -LiteralPath $sshConfig)) {
    throw "Missing H-drive SSH config: $sshConfig"
}
$sshOptions = @('-F', $sshConfig)

$localRoot = Join-Path $localControlRoot "returns\$JobId"
New-Item -ItemType Directory -Force $localRoot | Out-Null
$remoteSource = "$SshUser@$HostName`:$RemoteRoot/results/$JobId/"
& scp @sshOptions -r $remoteSource (Join-Path $localRoot '\')

$hashLines = Get-ChildItem -LiteralPath $localRoot -File -Recurse |
    Where-Object { $_.Name -ne 'retrieved_sha256sums.txt' } |
    ForEach-Object {
        $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
        $relative = [System.IO.Path]::GetRelativePath($localRoot, $_.FullName)
        "$($hash.Hash.ToLowerInvariant())  $relative"
    }
$hashLines | Sort-Object | Set-Content -Encoding UTF8 (Join-Path $localRoot 'retrieved_sha256sums.txt')
Write-Host "Retrieved remote result package to $localRoot"
