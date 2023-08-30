
代码参考 
- https://github.com/cambridgeltl/mop  
- https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA/tree/main/simpletransformers/seq2seq  
- https://github.com/shmsw25/bart-closed-book-qa

数据集来自 
- https://github.com/THU-KEG/KEPLER  
- https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA

cd data
wget -O wikidata5m_transductive.tar.gz https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1
wget -O wikidata5m_alias.tar.gz https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1
tar -xzvf wikidata5m_transductive.tar.gz
tar -xzvf wikidata5m_alias.tar.gz

问答数据集 https://drive.google.com/file/d/1K2uw9WXct6kA8i6_taJWeETGj2OdqgC1/view?usp=sharing


```bash
conda create -n qa python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
sudo wg-quick up tw
bash src/relation_prompt/run_pretrain.sh
bash src/evaluation/run_eval_wq.sh
bash src/evaluation/run_eval_nq.sh
bash src/evaluation/run_eval_triq.sh
```

