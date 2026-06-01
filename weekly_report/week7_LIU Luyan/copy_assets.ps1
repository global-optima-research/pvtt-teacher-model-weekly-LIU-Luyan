$base = 'D:\Learning_file\master&PhD\master\HKUST\毕设\IP-2026-spring\pvtt-teacher-model-weekly-LIU-Luyan'
$vp = "$base\VideoPainter"
$dst = "$base\weekly_report\week7_assets"

Copy-Item "$vp\experiments\exp1_pytorch_versions\pt2110_default\vis_mid.png" "$dst\exp1_pt2110_vis_mid.png" -Force
Copy-Item "$vp\experiments\exp0_colab_results\output_colab\vis_mid.png" "$dst\exp0_colab_vis_mid.png" -Force
Copy-Item "$vp\experiments\exp2_branch_ab_test\v3_scale1_mid.png" "$dst\exp2_branch_on.png" -Force
Copy-Item "$vp\experiments\exp2_branch_ab_test\v3_scale0_mid.png" "$dst\exp2_branch_off.png" -Force
Copy-Item "$vp\experiments\exp3_pure_cogvideox_i2v\v4_pure_i2v_mid.png" "$dst\exp3_pure_i2v.png" -Force
Copy-Item "$vp\experiments\exp4_no_offload\v6_no_offload_mid.png" "$dst\exp4_no_offload.png" -Force
Copy-Item "$vp\experiments\exp6_gradio_logic\results\frame_mid.png" "$dst\exp6_gradio_mid.png" -Force
Copy-Item "$vp\experiments\exp0_colab_results\output_colab\frame_mid.png" "$dst\exp0_colab_frame_mid.png" -Force
Write-Host 'Assets copied!'
