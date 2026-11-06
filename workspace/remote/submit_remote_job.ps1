[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [ValidateSet('dcu_cpu', 'user_gpu')]
    [string]$ExecutionTarget,
    [Parameter(Mandatory = $true)] [string]$HostName,
    [Parameter(Mandatory = $true)] [string]$SshUser,
    [Parameter(Mandatory = $true)] [string]$RemoteRoot,
    [ValidateSet('slurm', 'pbs', 'ssh')] [string]$Scheduler = 'slurm',
    [string]$Partition = '',
    [string]$JobId = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$projectRoot = $projectRoot.TrimEnd('\')
if ($projectRoot -notmatch '^[Hh]:\\') {
    throw "Refusing to stage outside the H: project tree: $projectRoot"
}
if ($Scheduler -eq 'ssh') {
    throw 'Direct ssh execution is disabled for numerical workloads; use Slurm or PBS.'
}
if ($ExecutionTarget -eq 'user_gpu') {
    throw 'user_gpu submission is blocked: no separately validated GPU particle backend is registered.'
}
if ([string]::IsNullOrWhiteSpace($Partition)) {
    throw 'A scheduler partition or queue is required for remote submission.'
}

$localControlRoot = 'external_workspaces/step30_remote_control_v1'
if (-not (Test-Path -LiteralPath $localControlRoot)) {
    New-Item -ItemType Directory -Force $localControlRoot | Out-Null
}
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

if ([string]::IsNullOrWhiteSpace($JobId)) {
    $JobId = "step30_${ExecutionTarget}_$(Get-Date -Format yyyyMMdd_HHmmss)"
}

$manifestPath = Join-Path $PSScriptRoot 'remote_job_manifest_v1.json'
$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
$manifest.target = $ExecutionTarget
$manifest.scheduler.kind = $Scheduler
$manifest.scheduler.partition_or_queue = $Partition
$manifest.connection.host = $HostName
$manifest.connection.ssh_user = $SshUser
$manifest.connection.remote_root = $RemoteRoot
$manifest.result.job_id = $JobId
$manifest.result.status = 'staged_not_submitted'
$manifestPathForJob = Join-Path $localControlRoot "manifests\remote_job_manifest_$JobId.json"
$manifestParent = Split-Path -Parent $manifestPathForJob
New-Item -ItemType Directory -Force $manifestParent | Out-Null
$manifest | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $manifestPathForJob

$remoteStage = "$SshUser@$HostName`:$RemoteRoot"
& ssh @sshOptions "$SshUser@$HostName" "mkdir -p '$RemoteRoot' '$RemoteRoot/results/$JobId'"
& scp @sshOptions -r (Join-Path $projectRoot 'workspace') "$remoteStage/"
& scp @sshOptions $manifestPathForJob "$remoteStage/remote_job_manifest_$JobId.json"

if ($Scheduler -eq 'slurm') {
    $remoteCommand = "cd '$RemoteRoot' && export STEP30_EXECUTION_TARGET=$ExecutionTarget STEP30_REMOTE_EXECUTION=1 STEP30_REMOTE_ROOT='$RemoteRoot' STEP30_JOB_ID='$JobId' && sbatch --parsable --export=ALL --job-name='$JobId' --partition='$Partition' workspace/remote/dcu_cpu_particle_confirmation_v2.sbatch"
} elseif ($Scheduler -eq 'pbs') {
    $remoteCommand = "cd '$RemoteRoot' && export STEP30_EXECUTION_TARGET=$ExecutionTarget STEP30_REMOTE_EXECUTION=1 STEP30_REMOTE_ROOT='$RemoteRoot' STEP30_JOB_ID='$JobId' && qsub -N '$JobId' workspace/remote/dcu_cpu_particle_confirmation_v2.sbatch"
}

Write-Host 'Staging complete. Submitting remotely; no numerical command runs locally.'
& ssh @sshOptions "$SshUser@$HostName" $remoteCommand
