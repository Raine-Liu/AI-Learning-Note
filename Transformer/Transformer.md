seq2seq: e.g.Speech Recognition, Machine Translation, Speech Translation
for example: 
	台语语音 -> 中文
	Text - to - Speech(TTS) Synthesis
	Chatbot: input -- seq2seq -- response
	Most Natural Language Possessing application
	QA can be done by seq2seq
	Syntactic Parsing: 句子结构 - 树 
	Multi - label Classification: An object can belong to multiple classes
	Object Detection

#### Seq2Seq
input sequence -- Encoder - > Decoder --output sequence
##### Encoder
![Transformer-1784021196334.webp](images/Transformer-1784021196334.webp)
residual connection
layer norm
![Transformer-1784021417365.webp](images/Transformer-1784021417365.webp)
**Encoder Block:**
![Transformer-1784021533935.webp](images/Transformer-1784021533935.webp)
*On Layer Normalization in the Transformer Architecture*
*PowerNorm*

##### Decoder
###### Autoregressive
以语音识别为例：
begin( 用one-hot表示）
V（输出范围）
得到的是一个概率分布     
![Transformer-1784028694429.webp](images/Transformer-1784028694429.webp)
将已有的输出再当作输入（one-hot)
![Transformer-1784028828594.webp](images/Transformer-1784028828594.webp)
会不会一步错，步步错？

Decoder:
![Transformer-1784028907439.webp](images/Transformer-1784028907439.webp)
**Masked Self-attention**
**Masked** Multi-Head Attention
	![Transformer-1784029150142.webp](images/Transformer-1784029150142.webp)
	![Transformer-1784029183338.webp](images/Transformer-1784029183338.webp)
	输出是一个一个产生的，只能考虑已输出的内容

输出什么时候停下来？准备END符号
![Transformer-1784029431158.webp](images/Transformer-1784029431158.webp)![Transformer-1784029479056.webp](images/Transformer-1784029479056.webp)

###### AT v.s. NAT
NAT 一次就可得到输出的句子
如何得到输出长度？
	另外设计预测输出长度的模型
	输出超长序列，舍弃end后的内容
NAT可以并行计算，输出长度可控
![Transformer-1784029896794.webp](images/Transformer-1784029896794.webp)
NAT is usually worse than AT (why? Multi-modality)

##### Encoder -- Decoder
Cross attention
![Transformer-1784030122047.webp](images/Transformer-1784030122047.webp)
![Transformer-1784030343998.webp](images/Transformer-1784030343998.webp)
cross attention 的诞生早于 self-attention (*Listen, attend and spell*)
在原始的论文中，将encoder最后一层的输出输入decorder，但后续有各种各样的变式

#### Train
类似分类机制
![Transformer-1784033887376.webp](images/Transformer-1784033887376.webp)
希望所有分类的cross entropy 尽量小
不要忘记END !
Teacher Forcing: 在训练时会给Decoder正确答案
![Transformer-1784034056018.webp](images/Transformer-1784034056018.webp)

##### Tips
###### Copy Mechanism
让decoder从encoder的输入中复制资料
	e.g.chat-bot可能遇到一些特定称谓、生成summary的model需要从原文中摘取词汇
*Pointer Network*
*Incorporation Copying Mechanism*

###### Guided Attention
TTS as example
	顿挫的产生和莫名的漏字
In some tasks, input and output are monotonically aligned. For example, speech recognition, TTS, etc.
有时我们要求机器在attention时有固定方式
	语音合成：需要从前往后注意
	在training中放置这样的注意方式
	*Monotonic Attention*
	*Location-aware Attention*

###### Beam Search
Assume there are only two tokens (V=2)
每次都会选分数高的那一个 -- Greedy Decoding
有时选了不太好的节点反而能得到最佳路径
![Transformer-1784034957241.webp](images/Transformer-1784034957241.webp)
遍历全局？-- 不现实
*Beam Search*
*The Curious Case of Neural Text Degeneration*  (Randomness is needed for decoder when generating sequence in some tasks e.g.,sentence completion, TTS)
训练时需要加入杂讯，测试时有时也需要加入杂讯
###### Optimizing Evaluation Metrics ？
训练时：minimize entropy
测试时：maximize BLEU score
没有对应
那训练时就maximize BLEU score ? -- Loss 无法微分
一种解决方案：用RL硬做

###### Scheduled Sampling
给Decoder的输入中故意加入错的值，避免输出时一步错步步错
