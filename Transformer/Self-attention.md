Sophisticated Input: Input is a set of vectors (may change length)
e.g. 文字处理：每个词汇是一个向量 
	文字--向量：one-hot encoding  --无法表示词汇间的相关性
	Word Embedding
一段声音信号：一排向量
	25ms -- frame, 10ms--window
一个Graph
	社交网络中将每个人的资讯看作向量
	一个分子看作一张图，原子为向量

What is the output?
- Each vector has a label (N--model--N)
	词性分类
	语音辨识
- The whole sequence has a label
	Sentiment analysis: 一个句子只要一个label
	辨析语音的speaker
- Model decides the number of labels itself (seq2seq)

#### Sequence Labeling
N to N
法一：将每个词都分别输入Fully-connected network
	不行，同一个词有不同的意思
法二：将附近几个词一起传入(window)
	有的长文本可能需要考虑远处的词
##### Self- attention
考虑了一整个句子的资讯
![Self-attention-1783939690494.webp](images/Self-attention-1783939690494.webp)![Self-attention-1783939733085.webp](images/Self-attention-1783939733085.webp)
![Self-attention-1783939805934.webp](images/Self-attention-1783939805934.webp)
怎么找到b1?
- Find the relevant vectors in a sequence
	判断哪些信息与a1有关，相关性如何
	计算模组：Dot-product
	![Self-attention-1783940025795.webp](images/Self-attention-1783940025795.webp)
	![Self-attention-1783940236114.webp](images/Self-attention-1783940236114.webp)
	$\alpha = q^i · k^j$
	softmax也可替换为其他激活函数
	![Self-attention-1783940500542.webp](images/Self-attention-1783940500542.webp)
	根据关联性抽取重要信息：谁的关联性更高就在b中占比更大，更起决定作用
	b1~b4是同时算出的
用矩阵表示：
将多个向量拼接可以得到矩阵：
![Self-attention-1784002810702.webp](images/Self-attention-1784002810702.webp)
![Self-attention-1784003032133.webp](images/Self-attention-1784003032133.webp)	![Self-attention-1784003174165.webp](images/Self-attention-1784003174165.webp)
总结：
![Self-attention-1784003377765.webp](images/Self-attention-1784003377765.webp)

##### Multi-head Self-attention
**Different types of relevance**
产生n种不同的head来找n种不同的相关性
每个head分别计算
![Self-attention-1784011329516.webp](images/Self-attention-1784011329516.webp)

##### Positional Encoding
No position information in self-attention :每个位置之间的距离都一样
如果有需要，我们需要位置信息
Each position has a unique positional vector $e^i$
hand-crafted? learn from data?
自己去探索吧

##### Implement of Self Attention
###### For Speech
Speech is a very long vector sequence.-> Attention Matrix L\*L complex
Truncated Self-attention: only care a range of data
![Self-attention-1784012026878.webp](images/Self-attention-1784012026878.webp)
###### For image
An image can also be considered as a vector set
![Self-attention-1784012172351.webp](images/Self-attention-1784012172351.webp)
**Self-attention v.s. CNN
![Self-attention-1784012266376.webp](images/Self-attention-1784012266376.webp)
*Paper:On the relationship between Self-Attention and Convolutional layers*
*Paper:An Image is Worth 16x16 Words: Transformer for Image Recognition at Scale*

**Self-attention v.s. RNN**:
![Self-attention-1784012914290.webp](images/Self-attention-1784012914290.webp)
*Paper：Transformer are RNNs*

###### Self-attention for Graph
Consider edge: only attention to connected nodes
只计算有边相连的node的attention分数 -> GNN
![Self-attention-1784013142287.webp](images/Self-attention-1784013142287.webp)

###### To Learn More
Long Range Arena: A benchmark for Efficient Transformer
Efficient Transformers: A survy