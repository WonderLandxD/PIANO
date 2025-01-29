




if __name__ == "__main__":
    checkpoint_path = "/mnt/sdb/ljw/opensdpc_codebase/weights/conch_v1_0.bin"
    model, image_preprocess, text_preprocess = create_pretrained_model('conch_v1_0', checkpoint_path, device='cuda')
    from PIL import Image
    image = Image.open('/mnt/sdb/ljw/opensdpc_codebase/patches/ljw_tissuenet_2025-01-04/C01_B008_S01/no000000_00007168x_00008064y.jpg')

    text = ["a image of tumor."]
    print('succefully load model.')
    model.set_mode('eval')
    image_tensor = image_preprocess(image).unsqueeze(0).cuda()  # [1, 3, 448, 448]
    text_tensor = text_preprocess(text).unsqueeze(0).cuda()  # [1, 77]
    # print(text_tensor.shape)
    
    # image_tensor = torch.randn(6, 3, 448, 448).cuda()
    img_feat = model.encode_image(image_tensor)
    text_feat = model.encode_text(text_tensor)
    print('succefully extract features.')
    print("image feature shape:", img_feat.shape, "text feature shape:", text_feat.shape)


'''
model.backbone is the raw model, preprocess is the valid preprocess (transform) function, model.forward is the image feature extraction function.
'''
