# Собирает указанные корневые .py и всю папку src в один текстовый файл с путями и разделителями.
# Исключения: массивы $excludeFolderNames и $excludeFileNames в теле скрипта.
# Использование: .\export_project_to_text.ps1
#                .\export_project_to_text.ps1 -OutFile "C:\path\export.txt"

[CmdletBinding()]
param(
    [string] $OutFile = "project_export.txt",
    [string] $ProjectRoot = ""
)

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($PSCommandPath) { Split-Path -Parent $PSCommandPath } else { (Get-Location).Path }
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

$ErrorActionPreference = "Stop"
$enc = [System.Text.UTF8Encoding]::new($false) # UTF-8 без BOM

# Paths in export: relative to project root, forward slashes only
function Get-ProjectRelativePath {
    param(
        [string] $FileFullPath
    )
    $base = $ProjectRoot.TrimEnd([char[]]@('\', '/'))
    $full = [System.IO.Path]::GetFullPath($FileFullPath)
    if ($full.Length -lt $base.Length -or -not $full.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is not under project root: $FileFullPath (root: $ProjectRoot)"
    }
    $tail = if ($full.Length -eq $base.Length) { "" } else { $full.Substring($base.Length).TrimStart([char[]]@('\', '/')) }
    if ([string]::IsNullOrEmpty($tail)) { return "." }
    return ($tail -replace '\\', '/')
}

$rootFiles = @(
    "bt.py",
    "bt_walk_forward.py",
    "etl.py",
    "paper.py",
    "train.py"
)

# --- Исключения при обходе: добавьте сюда ненужные папки и файлы ---

# Имя сегмента пути: если встречается в любом месте (как .git), каталог с содержимым пропускается.
$excludeFolderNames = @(
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "lstm",
    "lstm_candles",
    "sql"
    # "dist",
    # ".idea",
)

# Только имя файла (не путь). Регистр не важен. Поддержка wildcards: "*.log", "thumbs.db"
$excludeFileNames = @(
    # "local_secrets.json",
    # "package-lock.json"
)

# Неэкспортируемые расширения (бинарные и прочее)
$skipExtensions = @(".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".db", ".sqlite", ".png", ".jpg", ".ico")

function Test-IsExcludedFileName {
    param([string] $LeafName, [string[]] $Patterns)
    foreach ($p in $Patterns) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        if ($p.IndexOfAny([char[]]@('*', '?')) -ge 0) {
            if ($LeafName -like $p) { return $true }
        } elseif ($LeafName.Equals($p, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Test-IsExcludedPath {
    param([string] $FullPath, [string] $BaseRoot)
    $leaf = [System.IO.Path]::GetFileName($FullPath)
    if (Test-IsExcludedFileName -LeafName $leaf -Patterns $excludeFileNames) { return $true }

    $rel = $FullPath.Substring($BaseRoot.Length).TrimStart("\", "/")
    $parts = $rel -split "[\\/]"
    foreach ($d in $excludeFolderNames) {
        if ([string]::IsNullOrWhiteSpace($d)) { continue }
        if ($parts -contains $d) { return $true }
    }
    $ext = [System.IO.Path]::GetExtension($FullPath)
    if ($skipExtensions -contains $ext) { return $true }
    return $false
}

$sb = [System.Text.StringBuilder]::new()
$null = $sb.AppendLine("================================================================")
$null = $sb.AppendLine("PROJECT TEXT EXPORT")
$null = $sb.AppendLine("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$null = $sb.AppendLine("Root: .")
$null = $sb.AppendLine("(All FILE: paths below are relative to the project root.)")
$null = $sb.AppendLine("================================================================")
$null = $sb.AppendLine()

# Корневые файлы
foreach ($name in $rootFiles) {
    if (Test-IsExcludedFileName -LeafName $name -Patterns $excludeFileNames) { continue }
    $p = Join-Path $ProjectRoot $name
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        Write-Warning "File not found, skip: $p"
        continue
    }
    $rel = Get-ProjectRelativePath -FileFullPath $p
    $null = $sb.AppendLine("----------------------------------------------------------------")
    $null = $sb.AppendLine("FILE: $rel")
    $null = $sb.AppendLine("----------------------------------------------------------------")
    $content = [System.IO.File]::ReadAllText($p, $enc)
    $null = $sb.AppendLine($content)
    if (-not $content.EndsWith("`n")) { $null = $sb.AppendLine() }
    $null = $sb.AppendLine("----------------------------------------------------------------")
    $null = $sb.AppendLine("END FILE: $rel")
    $null = $sb.AppendLine()
}

# Папка src
$srcPath = Join-Path $ProjectRoot "src"
if (-not (Test-Path -LiteralPath $srcPath -PathType Container)) {
    Write-Warning "Folder not found: $srcPath"
} else {
    $all = Get-ChildItem -LiteralPath $srcPath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-IsExcludedPath -FullPath $_.FullName -BaseRoot $ProjectRoot) } |
        Sort-Object FullName

    foreach ($f in $all) {
        $rel = Get-ProjectRelativePath -FileFullPath $f.FullName
        $null = $sb.AppendLine("----------------------------------------------------------------")
        $null = $sb.AppendLine("FILE: $rel")
        $null = $sb.AppendLine("----------------------------------------------------------------")
        try {
            $content = [System.IO.File]::ReadAllText($f.FullName, $enc)
        } catch {
            Write-Warning "UTF-8 read failed, using system default: $rel"
            $content = [System.IO.File]::ReadAllText($f.FullName, [System.Text.Encoding]::Default)
        }
        $null = $sb.AppendLine($content)
        if (-not $content.EndsWith("`n")) { $null = $sb.AppendLine() }
        $null = $sb.AppendLine("----------------------------------------------------------------")
        $null = $sb.AppendLine("END FILE: $rel")
        $null = $sb.AppendLine()
    }
}

$outPath = if ([System.IO.Path]::IsPathRooted($OutFile)) { $OutFile } else { Join-Path $ProjectRoot $OutFile }
[System.IO.File]::WriteAllText($outPath, $sb.ToString(), $enc)
Write-Host "Done: $outPath"
