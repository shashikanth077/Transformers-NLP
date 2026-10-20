# Opens the generated .docx in Microsoft Word, updates the table of contents and all fields,
# saves it, and exports a PDF copy. Usage: powershell -File word_finalize.ps1 <docx> <pdf>
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)
    $stats = $doc.ComputeStatistics(2)   # pages
    $chars = $doc.ComputeStatistics(5)   # characters with spaces
    $nospace = $doc.ComputeStatistics(3) # characters without spaces
    $words = $doc.ComputeStatistics(0)
    Write-Output "pages=$stats words=$words chars_with_spaces=$chars chars_no_spaces=$nospace"
    $doc.Close()
} finally {
    $word.Quit()
}
