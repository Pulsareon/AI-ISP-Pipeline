import os
import time
import cv2
import numpy as np
import onnxruntime as ort

def create_dummy_model(path):
    import onnx
    from onnx import helper
    from onnx import TensorProto

    # 创建一个简单的恒等变换模型 (Identity)
    # 输入: (1, 1, 512, 512), 输出: (1, 1, 512, 512)
    input_node = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 512, 512])
    output_node = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 512, 512])

    node_def = helper.make_node(
        'Identity',
        ['input'],
        ['output'],
    )

    graph_def = helper.make_graph(
        [node_def],
        'dummy-model',
        [input_node],
        [output_node],
    )

    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    onnx.save(model_def, path)
    print(f"已生成虚拟模型: {path}")

def main():
    model_path = "model.onnx"
    input_path = "input_bayer.png"
    output_path = "output.png"

    # 1. 检查或生成随机 Bayer 图
    if not os.path.exists(input_path):
        print(f"未找到输入图片，正在生成随机 Bayer 模拟图: {input_path}")
        # 生成 512x512 的单通道随机噪声图作为 Bayer 输入
        random_bayer = np.random.randint(0, 256, (512, 512, 1), dtype=np.uint8)
        cv2.imwrite(input_path, random_bayer)

    # 读取图片
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"错误: 无法读取图片 {input_path}")
        return

    # 确保是 3D (H, W, C)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    print(f"输入尺寸: {img.shape}")
    
    # 2. 检查模型是否存在，不存在则尝试生成一个 Dummy 模型 (需要 onnx 库)
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型文件 '{model_path}'。")
        try:
            create_dummy_model(model_path)
        except ImportError:
            print("错误: 缺少 'onnx' 库，无法自动生成虚拟模型。请提供一个有效的 model.onnx")
            return
        except Exception as e:
            print(f"生成虚拟模型失败: {e}")
            return

    # 初始化推理会话
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"推理会话初始化失败: {e}")
        return

    input_name = session.get_inputs()[0].name
    print(f"模型输入节点名称: {input_name}")

    # 3. 预处理
    # 归一化 [0, 1]
    input_data = img.astype(np.float32) / 255.0
    # 维度调整: HWC -> NCHW
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    # 4. 推理
    print("正在执行推理...")
    start_time = time.perf_counter()
    try:
        outputs = session.run(None, {input_name: input_data})
    except Exception as e:
        print(f"推理过程中出错: {e}")
        return
    end_time = time.perf_counter()
    
    print(f"推理耗时: {(end_time - start_time) * 1000:.2f} ms")

    # 5. 后处理
    output_tensor = outputs[0]
    # NCHW -> HWC
    output_img = np.squeeze(output_tensor, axis=0)
    output_img = np.transpose(output_img, (1, 2, 0))
    
    # 反归一化并限制在 [0, 255]
    output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)

    # 6. 保存结果
    cv2.imwrite(output_path, output_img)
    print(f"处理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
