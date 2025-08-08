from transformers import PretrainedConfig


class Olmo2Config(PretrainedConfig):
    """
    Configuration class for OLMo-2 model.
    """
    
    model_type = "olmo2"
    
    def __init__(
        self,
        vocab_size=100352,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=100277,
        eos_token_id=100257,
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=4134231.132028111,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        
        super().__init__(**kwargs)
