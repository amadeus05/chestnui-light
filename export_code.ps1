$outputFile = "project_export.txt"
$files = @("etl.py", "config.py", "train.py", "bt.py")

if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

foreach ($file in $files) {
    if (Test-Path $file) {
        $fullPath = (Get-Item $file).FullName
        Add-Content -Path $outputFile -Value "================================================================================"
        Add-Content -Path $outputFile -Value "File: $fullPath"
        Add-Content -Path $outputFile -Value "================================================================================"
        Add-Content -Path $outputFile -Value ""
        
        $content = Get-Content $file -Raw
        if ($content) {
            Add-Content -Path $outputFile -Value $content
        }
        Add-Content -Path $outputFile -Value "`r`n`r`n"
        
        Write-Host "Exported $file"
    } else {
        Write-Host "Warning: File $file not found." -ForegroundColor Yellow
    }
}

Write-Host "Done! Exported project files to $($((Get-Item $outputFile).FullName))" -ForegroundColor Green
