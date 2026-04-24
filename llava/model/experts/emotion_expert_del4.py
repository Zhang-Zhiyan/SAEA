import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, q_embed_dim, kv_embed_dim, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        
                           
        self.query_proj = nn.Linear(q_embed_dim, embed_dim)
        self.key_proj = nn.Linear(kv_embed_dim, embed_dim)
        
    def forward(self, queries, keys, values):
        """
        queries: [batch_size, q_seq_len, embed_dim]
        keys: [batch_size, kv_seq_len, embed_dim]
        values: [batch_size, kv_seq_len, embed_dim]
        """
                        
        Q = self.query_proj(queries)                                      
        K = self.key_proj(keys)                                            
        V = values                                                          
        
                                   
        d_k = K.shape[-1]             
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
                                
        attention_weights = F.softmax(attention_scores, dim=-1)                                       
        
                        
        attention_output = torch.matmul(attention_weights, V)                                         
        
        return attention_output, attention_weights


class EmotionOV(nn.Module):
    def __init__(self, embedding_dim=512, class_dim=512, kernal_size=3, attention_dim=512, num_pooling_tokens=2, num_emotions=9):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.class_dim = class_dim
        self.kernal_size = kernal_size
        self.attention_dim = attention_dim
        self.num_pooling_tokens = num_pooling_tokens
        self.num_emotions = num_emotions          

        self.num_patches = 729
        self.ov_token_hidden_size = 1152

        self._build_emotion_components()
        self._build_q_former()
     

    def _build_emotion_components(self):
                               
                                                               
        self.emotion_embeddings = nn.Parameter(
            torch.randn(self.num_emotions, self.embedding_dim),
            requires_grad=True
        )
                                                                                                                        
        self.pooling_layer = nn.Sequential(
                                nn.Linear(self.num_patches, 1),
                            )
        
                                                                                                
        self.emotion_classifier = nn.Sequential(
                                        nn.Linear(self.ov_token_hidden_size, self.class_dim),
                                        nn.GELU(),
                                        nn.Linear(self.class_dim, self.class_dim),
                                    )
        
                            
        self.siglip_adaptors = nn.ModuleList([
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
        ])

        self.emotion_prompt_adaptors = nn.ModuleList([
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
        ])

        self.final_emotion_predictor = nn.Sequential(
                                    nn.Linear(self.ov_token_hidden_size, 512),
                                    nn.GELU(),
                                    nn.Linear(512, self.num_emotions),
                                )
        
        self.global_pooling = nn.AdaptiveAvgPool2d((self.num_pooling_tokens, self.num_pooling_tokens))


    def _build_q_former(self):
        self.q_former = SingleHeadAttention(self.ov_token_hidden_size, self.ov_token_hidden_size, self.attention_dim)

    def freeze_layers(self):
                                   
        for param in self.parameters():
            param.requires_grad = False

    def load_zero_shot_weights(self, path='./pretrained_expert_7b.pth'):
        """加载预训练的情感专家权重"""
        checkpoint = torch.load(path)
        self_state_dict = self.state_dict()
        matched_names = []
        for key in self_state_dict.keys():
            if key in checkpoint.keys():
                matched_names.append(key)

        print(f"Matched names: {matched_names}")
        print('Loaded number of keys:', len(matched_names))
        self.load_state_dict(checkpoint, strict=False)

    def get_emotion_map(self, sig_multi_level_features, ov_base_features): 
        """
        生成情感注意力图，显示图像中与情感最相关的区域

        Args:
            sig_multi_level_features: list of tensors, 每个tensor是[batch_size * num_patches, 729, siglip_hidden_dim]
            ov_base_features: tensor [batch_size * num_patches, 729, ov_token_hidden_size]

        Returns:
            emotion_attention_map: [batch_size * num_patches, num_emotions, 27, 27]
            patch_emotion_scores: [batch_size * num_patches, num_emotions]
        """
        total_size = ov_base_features.shape[0]

                
        emotion_embeddings = self.emotion_embeddings.unsqueeze(0).repeat(total_size, 1, 1)                                                           

                     
        ov_base_features_pooled = self.pooling_layer(ov_base_features.transpose(1, 2)).squeeze(-1)                                                    
        class_features = self.emotion_classifier(ov_base_features_pooled)                                         

                       
        emotion_prompt_embeddings_list = []
        for i in range(4):        
            emotion_prompts = []
            for emotion_idx in range(self.num_emotions):
                emotion_emb = emotion_embeddings[:, emotion_idx, :]                                             
                emotion_prompt = self.emotion_prompt_adaptors[i](
                    torch.cat([emotion_emb, class_features], dim=-1)
                )                                             
                emotion_prompt = emotion_prompt / emotion_prompt.norm(dim=-1, keepdim=True)
                emotion_prompts.append(emotion_prompt.unsqueeze(1))                                                

            emotion_prompt_embeddings = torch.cat(emotion_prompts, dim=1)                                                           
            emotion_prompt_embeddings_list.append(emotion_prompt_embeddings)

                         
        sig_emotion_embeddings_list = []
        for i in range(4):
            sig_embedding = self.siglip_adaptors[i](sig_multi_level_features[i])                                                  
            sig_embedding = sig_embedding / sig_embedding.norm(dim=-1, keepdim=True)
            sig_emotion_embeddings_list.append(sig_embedding)

                  
        emotion_attention_maps = []
        patch_emotion_scores_list = []
        for i in range(4):
                     
            attention_scores = torch.matmul(
                100.0 * sig_emotion_embeddings_list[i],
                emotion_prompt_embeddings_list[i].transpose(-2, -1)
            )                                                 

                     
            attention_map = attention_scores.permute(0, 2, 1).view(
                total_size, self.num_emotions, 27, 27
            )                                                    

                              
            attention_map = torch.softmax(attention_map.view(total_size, self.num_emotions, -1), dim=-1)
            attention_map = attention_map.view(total_size, self.num_emotions, 27, 27)

            emotion_attention_maps.append(attention_map)

                            
            patch_scores = attention_map.mean(dim=(2, 3))                                            
            patch_emotion_scores_list.append(patch_scores)

                   
        final_emotion_attention_map = sum(emotion_attention_maps) / 4                                                    
        final_patch_emotion_scores = sum(patch_emotion_scores_list) / 4                                            

                         
        self.last_attention_maps = final_emotion_attention_map

        return final_emotion_attention_map, final_patch_emotion_scores
    
    def forward(self, ov_image_features, sig_multi_level_features, split_sizes):
        """
        情感专家前向传播

        Args:
            ov_image_features: [batch_size * num_patches, 729, ov_token_hidden_size]
            sig_multi_level_features: list of tensors, 每个tensor是[batch_size * num_patches, 729, siglip_hidden_dim]
            split_sizes: list of integers, 每个图像的patch数量

        Returns:
            emotion_features: [batch_size * num_patches, 4, ov_token_hidden_size] - 情感特征
            global_emotion_features: [batch_size, 1, ov_token_hidden_size] - 全局情感特征
            emotion_predictions: [batch_size, num_emotions] - 情感概率分布
        """
        batch_size = len(split_sizes)
        ov_image_features_split = torch.split(ov_image_features, split_sizes)
        ov_base_image_feature_list = []

        for i, image_feature in enumerate(ov_image_features_split):
            base_image_feature = image_feature[0].unsqueeze(0)                                  
            ov_base_image_feature_list.append(base_image_feature.repeat(split_sizes[i], 1, 1))

        ov_base_image_features = torch.cat(ov_base_image_feature_list, dim=0)                                                         

                  
        emotion_attention_map, patch_emotion_scores = self.get_emotion_map(
            sig_multi_level_features, ov_base_image_features
        )                                                    

                      
        ov_image_features_reshaped = ov_image_features.permute(0, 2, 1).view(
            ov_image_features.shape[0], self.ov_token_hidden_size, 27, 27
        )                                                            

                                  
        max_emotion_attention = emotion_attention_map.max(dim=1, keepdim=True)[0]                                         

        scaled_emotion_features = ov_image_features_reshaped * max_emotion_attention                                                            
        emotion_significance = self.global_pooling(max_emotion_attention)                                       
        emotion_tokens = self.global_pooling(scaled_emotion_features) / (emotion_significance + 1e-8)                                                          
        emotion_significance = emotion_significance.permute(0, 2, 3, 1).view(
            scaled_emotion_features.shape[0], self.num_pooling_tokens * self.num_pooling_tokens, 1
        )                                    

                   
        global_emotion_significance = emotion_significance * patch_emotion_scores.max(dim=-1, keepdim=True)[0].unsqueeze(-1)                                    

                    
        emotion_tokens = emotion_tokens.permute(0, 2, 3, 1).view(
            scaled_emotion_features.shape[0], self.num_pooling_tokens * self.num_pooling_tokens, self.ov_token_hidden_size
        )                                                       

                           
        major_emotion_output, _ = self.q_former(emotion_tokens, ov_image_features, ov_image_features)                                                       

                   
        global_emotion_significance_split = torch.split(global_emotion_significance, split_sizes)
        emotion_output_split = torch.split(major_emotion_output, split_sizes)
        emotion_output_list = []

        for i in range(batch_size):
            emotion_output = (emotion_output_split[i] * global_emotion_significance_split[i]).mean(dim=(0, 1)) / (
                global_emotion_significance_split[i].mean(dim=(0, 1)) + 1e-8
            )                           
            emotion_output_list.append(emotion_output.unsqueeze(0))                             

        final_emotion_output = torch.cat(emotion_output_list, dim=0).unsqueeze(1)                                         

                  
        emotion_predictions = self.final_emotion_predictor(final_emotion_output.squeeze(1))                              
        emotion_predictions = torch.softmax(emotion_predictions, dim=-1)                                     

        return major_emotion_output, final_emotion_output, emotion_predictions
