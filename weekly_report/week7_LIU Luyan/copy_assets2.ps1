$vp = Join-Path $PSScriptRoot '..\VideoPainter'
$dst = Join-Path $PSScriptRoot 'week7_assets'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$copies = @(
    @((Join-Path $vp 'experiments\exp1_pytorch_versions\pt2110_default\vis_mid.png'), (Join-Path $dst 'exp1_vis_mid.png')),
    @((Join-Path $vp 'experiments\exp0_colab_results\output_colab\vis_mid.png'), (Join-Path $dst 'exp0_colab_vis_mid.png')),
    @((Join-Path $vp 'experiments\exp0_colab_results\output_colab\frame_mid.png'), (Join-Path $dst 'exp0_colab_frame_mid.png')),
    @((Join-Path $vp 'experiments\exp2_branch_ab_test\v3_scale1_mid.png'), (Join-Path $dst 'exp2_branch_on.png')),
    @((Join-Path $vp 'experiments\exp2_branch_ab_test\v3_scale0_mid.png'), (Join-Path $dst 'exp2_branch_off.png')),
    @((Join-Path $vp 'experiments\exp3_pure_cogvideox_i2v\v4_pure_i2v_mid.png'), (Join-Path $dst 'exp3_pure_i2v.png')),
    @((Join-Path $vp 'experiments\exp4_no_offload\v6_no_offload_mid.png'), (Join-Path $dst 'exp4_no_offload.png')),
    @((Join-Path $vp 'experiments\exp5_with_lora\v7_lora_mid.png'), (Join-Path $dst 'exp5_lora.png')),
    @((Join-Path $vp 'experiments\exp6_gradio_logic\results\frame_mid.png'), (Join-Path $dst 'exp6_gradio.png'))
)
foreach ($c in $copies) {
    if (Test-Path $c[0]) { Copy-Item $c[0] $c[1] -Force; Write-Host "OK: $($c[1])" }
    else { Write-Host "MISSING: $($c[0])" }
}
