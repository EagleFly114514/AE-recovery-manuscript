# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
param(
    [string]$Inputs,
    [string]$Out,
    [ValidateSet('clinical','primary','precision','multiple','icu','legacy','posthoc','tables','figures','all','smoke','check-code')]
    [string]$Stage='smoke',
    [string]$Python='python',
    [int]$Jobs=8,
    [string]$SavedPredictions
)
$ErrorActionPreference='Stop'
if(-not $Out -and $Stage -ne 'check-code'){throw 'Specify a separate output directory with -Out.'}
$taskRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskArgs=@((Join-Path $taskRoot 'reproduce.py'),'--stage',$Stage,'--jobs',$Jobs)
if($Out){$taskArgs+=@('--out',$Out)}
if($Inputs){$taskArgs+=@('--inputs',$Inputs)}
if($SavedPredictions){$taskArgs+=@('--saved-predictions',$SavedPredictions)}
& $Python @taskArgs
if($LASTEXITCODE -ne 0){throw ('Reproduction failed with exit code: '+$LASTEXITCODE)}
