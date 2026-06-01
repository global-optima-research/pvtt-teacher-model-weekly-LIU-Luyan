$base = $PSScriptRoot

# Create folders
$dirs = @('experiments\exp1_pytorch_versions','experiments\exp2_branch_ab_test','experiments\exp3_pure_cogvideox_i2v','experiments\exp4_no_offload','experiments\exp5_with_lora','experiments\exp6_gradio_logic','experiments\exp0_colab_results','test_scripts','notebooks','reports')
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $base $d) | Out-Null }

# Move version folders
$folderMoves = @(
    @('20260405版本','experiments\exp1_pytorch_versions\pt2110_default'),
    @('20260405_nocudnn版本','experiments\exp1_pytorch_versions\pt2110_nocudnn'),
    @('20260405_pt280版本','experiments\exp1_pytorch_versions\pt280'),
    @('20260405_compile版本','experiments\exp1_pytorch_versions\pt280_compile'),
    @('output_colab_v0','experiments\exp0_colab_results\output_colab'),
    @('output_official_v0','experiments\exp0_colab_results\output_official'),
    @('exp8_gradio_result','experiments\exp6_gradio_logic\results')
)
foreach ($m in $folderMoves) {
    $src = Join-Path $base $m[0]
    $dst = Join-Path $base $m[1]
    if (Test-Path $src) { Move-Item $src $dst -Force }
}

# Move files
$fileMoves = @(
    @('branch_ON_mid.png','experiments\exp2_branch_ab_test'),
    @('branch_OFF_mid.png','experiments\exp2_branch_ab_test'),
    @('v3_scale0_mid.png','experiments\exp2_branch_ab_test'),
    @('v3_scale1_mid.png','experiments\exp2_branch_ab_test'),
    @('v4_pure_i2v_mid.png','experiments\exp3_pure_cogvideox_i2v'),
    @('v6_no_offload_mid.png','experiments\exp4_no_offload'),
    @('v7_lora_mid.png','experiments\exp5_with_lora'),
    @('debug_report_0407.md','reports'),
    @('debug_session_notes.md','reports'),
    @('LED_grid_0316.md','reports'),
    @('VideoPainter_Colab.ipynb','notebooks'),
    @('VideoPainter_Colab_Official.ipynb','notebooks')
)
foreach ($m in $fileMoves) {
    $src = Join-Path $base $m[0]
    $dst = Join-Path $base $m[1]
    if (Test-Path $src) { Move-Item $src $dst -Force }
}

# Move test scripts
Get-ChildItem (Join-Path $base 'test_exp*.py') -ErrorAction SilentlyContinue | Move-Item -Destination (Join-Path $base 'test_scripts') -Force

Write-Host 'Done!'
