param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf,
    [Parameter(Mandatory=$true)][string]$AuditJson
)
$ErrorActionPreference = 'Stop'
$inputFull = (Resolve-Path -LiteralPath $InputDocx).Path
$outputFull = [System.IO.Path]::GetFullPath($OutputPdf)
$auditFull = [System.IO.Path]::GetFullPath($AuditJson)
if ([System.IO.Path]::GetExtension($inputFull) -ne '.docx') { throw 'Input must be DOCX.' }
if ([System.IO.Path]::GetExtension($outputFull) -ne '.pdf') { throw 'Output must be PDF.' }
if (Test-Path -LiteralPath $outputFull) { throw 'Refusing to overwrite existing PDF.' }
if (Test-Path -LiteralPath $auditFull) { throw 'Refusing to overwrite existing audit.' }
if (-not (Test-Path -LiteralPath ([System.IO.Path]::GetDirectoryName($outputFull)))) { throw 'PDF parent directory must exist.' }
$sourceHash = (Get-FileHash -LiteralPath $inputFull -Algorithm SHA256).Hash
$word = $null
$document = $null
$ownApplication = $false
try {
    $word = New-Object -ComObject Word.Application
    if ($word.Documents.Count -ne 0) { throw 'Automation did not create an empty Word instance; leaving existing documents alone.' }
    $ownApplication = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    # Read-only; do not add to recent files or modify the DOCX.
    $document = $word.Documents.Open($inputFull, $false, $true, $false)
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    # Native Word PDF conversion: all pages, print quality, no document properties.
    $document.ExportAsFixedFormat($outputFull, 17, $false, 0, 0, 1, 1, 0, $false)
    $afterHash = (Get-FileHash -LiteralPath $inputFull -Algorithm SHA256).Hash
    if ($sourceHash -ne $afterHash) { throw 'Source DOCX unexpectedly changed.' }
    $audit = [ordered]@{
        renderer = 'Microsoft Word COM'; word_version = [string]$word.Version
        source = $inputFull; pdf = $outputFull; pages = [int]$pages
        source_sha256 = $sourceHash.ToLowerInvariant(); source_unchanged = $true
        include_document_properties = $false; opened_readonly = $true
        bytes = (Get-Item -LiteralPath $outputFull).Length
    }
    [System.IO.File]::WriteAllText($auditFull, ($audit | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
    $audit | ConvertTo-Json -Depth 5
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        if ($ownApplication) { $word.Quit(0) }
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
