from robokit.datasets.tools import T5EmbeddingsH5


# 1) 批量转换
save_to_dir = "/home/geyuan/datasets/ipec_lerobot/lang_emb_t5xxl/"
mapping = T5EmbeddingsH5.batch_convert_dir(save_to_dir, overwrite=False)
print("Converted:", mapping)

# 2) 打开某个 h5 做读取/追加
one_h5 = next(iter(mapping.values()))
with T5EmbeddingsH5(one_h5, mode="a") as store:
    print("N =", store.n_unique)

    # 构建索引并查 id
    store.build_text_index()
    some_text = "demo"
    idx = store.get_id(some_text)
    print("id =", idx)

    # 读取
    if idx is not None:
        t, emb, vlen = store.get_by_id(idx)
        print(t, emb.shape, vlen)

    # 追加（如果要去重：先 get_id 判断）
    # new_id = store.append_one("new text", np.zeros((512,1024), np.float16), 10)
    # print("new_id =", new_id)
