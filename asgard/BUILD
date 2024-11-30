load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "librknnrt_linux.1.6.0",
    srcs = ["lib/linux/librknnrt.so.1.6.0"],
)

cc_library(
    name = "librknnrt_android.1.6.0",
    srcs = ["lib/android/librknnrt.so.1.6.0"],
)

cc_binary(
    name = "guest_inference_tsdp",
    srcs = [
        "guest_inference_tsdp.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "guest_inference_minimal",
    tags = ["no_android"],
    srcs = [
        "guest_inference.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
    deps = [
        "librknnrt_linux.1.6.0",
    ],
)

cc_binary(
    name = "guest_inference",
    srcs = [
        "guest_inference.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
    deps = [
        "librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "host_inference",
    srcs = [
        "host_inference.cc",
        "include/stb/stb_image.h",
        "include/stb/stb_image_resize.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
)

cc_binary(
    name = "host_inference_lite_transformer",
    srcs = [
        "host_inference_lite_transformer.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
)

cc_binary(
    name = "host_inference_tsdp",
    srcs = [
        "host_inference_tsdp.cc",
        "include/rknn/1.6.0/rknn_api.h",
        "include/stb/stb_image.h",
        "include/stb/stb_image_resize.h",
    ],
    deps = [
        "librknnrt_android.1.6.0",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
)

cc_binary(
    name = "host_inference_native",
    srcs = [
        "host_inference_native.cc",
        "include/rknn/1.6.0/rknn_api.h",
        "include/stb/stb_image.h",
        "include/stb/stb_image_resize.h",
    ],
    deps = [
        "librknnrt_android.1.6.0",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",
        ],
    }),
)

cc_binary(
    name = "guest_inference_ssd_mobilenetv1_coalesced",
    srcs = [
        "guest_inference_ssd_mobilenetv1_coalesced.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "guest_inference_ssd_inceptionv2_coalesced",
    srcs = [
        "guest_inference_ssd_inceptionv2_coalesced.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "guest_inference_lite_transformer_decoder_coalesced",
    srcs = [
        "guest_inference_lite_transformer_decoder_coalesced.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "guest_inference_lite_transformer_encoder_coalesced",
    srcs = [
        "guest_inference_lite_transformer_encoder_coalesced.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "host_inference_native_ssd_mobilenetv1_baseline",
    srcs = [
        "host_inference_native_ssd_mobilenetv1_baseline.cc",
        "include/rknn/1.6.0/rknn_api.h",
        "include/stb/stb_image.h",
        "include/stb/stb_image_resize.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "host_inference_native_ssd_inceptionv2_baseline",
    srcs = [
        "host_inference_native_ssd_inceptionv2_baseline.cc",
        "include/rknn/1.6.0/rknn_api.h",
        "include/stb/stb_image.h",
        "include/stb/stb_image_resize.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "host_inference_native_lite_transformer_encoder_baseline",
    srcs = [
        "host_inference_native_lite_transformer_encoder_baseline.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "host_inference_native_lite_transformer_decoder_baseline",
    srcs = [
        "host_inference_native_lite_transformer_decoder_baseline.cc",
        "include/rknn/1.6.0/rknn_api.h",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        ":librknnrt_android.1.6.0",
    ],
)

cc_binary(
    name = "vm_service",
    srcs = [
        "vm_service.c",
    ],
)
