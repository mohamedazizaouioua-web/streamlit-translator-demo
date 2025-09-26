import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# PAGE CONFIGURATION
# Set the page title, icon, and layout. This is the first command to run.
# ==============================================================================
st.set_page_config(
    page_title="AI French Translator",
    page_icon="🇫🇷",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="expanded"
)

# ==============================================================================
# BACKEND: MODEL AND TOKENIZER LOADING (UNCHANGED)
# All the complex TensorFlow code remains the same.
# ==============================================================================
ASSETS_DIR = 'my_translation_model_assets'
MODEL_PATH = os.path.join(ASSETS_DIR, 'transformer_model.keras')
TOKENIZER_ENG_PATH = os.path.join(ASSETS_DIR, 'eng_tokenizer.pkl')
TOKENIZER_FRE_PATH = os.path.join(ASSETS_DIR, 'fre_tokenizer.pkl')
MAX_LENGTH = 40

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]); angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]; return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads, self.d_model = num_heads, d_model
        assert d_model % self.num_heads == 0; self.depth = d_model // self.num_heads
        self.wq, self.wk, self.wv, self.dense = [tf.keras.layers.Dense(d_model) for _ in range(4)]
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]; q, k, v = self.wq(q), self.wk(k), self.wv(v)
        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth)); k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth)); v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3]); k = tf.transpose(k, perm=[0, 2, 1, 3]); v = tf.transpose(v, perm=[0, 2, 1, 3])
        matmul_qk = tf.matmul(q, k, transpose_b=True); dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None: scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1); output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3]); concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(concat_attention), attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads); self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6); self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate); self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, x, *, training, mask):
        attn_output, _ = self.mha(x, x, x, mask); attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output); ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training); return self.layernorm2(out1 + ffn_output)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model, num_heads); self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6); self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6); self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate); self.dropout3 = tf.keras.layers.Dropout(rate)
    def call(self, x, *, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_w1 = self.mha1(x, x, x, look_ahead_mask); attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x); attn2, attn_w2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training); out2 = self.layernorm2(attn2 + out1); ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training); return self.layernorm3(ffn_output + out2), attn_w1, attn_w2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate=0.1, **kwargs):
        super().__init__(**kwargs); self.d_model = d_model; self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model); self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]; self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, *, training, mask):
        seq_len = tf.shape(x)[1]; x = self.embedding(x); x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]; x = self.dropout(x, training=training)
        for i in range(self.num_layers): x = self.enc_layers[i](x, training=training, mask=mask)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_seq_len, rate=0.1, **kwargs):
        super().__init__(**kwargs); self.d_model = d_model; self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model); self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]; self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, *, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]; attention_weights = {}; x = self.embedding(x); x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]; x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output=enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1; attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output=enc_output, training=training, look_ahead_mask=combined_mask, padding_mask=dec_padding_mask)
        return self.final_layer(dec_output), attention_weights
    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp); dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1]); dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask); return enc_padding_mask, combined_mask, dec_padding_mask
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32); return seq[:, tf.newaxis, tf.newaxis, :]
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0); return mask

@st.cache_resource
def load_assets():
    with open(TOKENIZER_ENG_PATH, 'rb') as f: eng_tokenizer = pickle.load(f)
    with open(TOKENIZER_FRE_PATH, 'rb') as f: fre_tokenizer = pickle.load(f)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    fre_vocab_size = len(fre_tokenizer.word_index) + 1
    num_layers, d_model, dff, num_heads, dropout_rate = 4, 128, 512, 8, 0.1
    transformer = Transformer(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        input_vocab_size=eng_vocab_size, target_vocab_size=fre_vocab_size,
        pe_input=MAX_LENGTH, pe_target=MAX_LENGTH, rate=dropout_rate)
    dummy_input = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    dummy_target = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    _ = transformer((dummy_input, dummy_target), training=False)
    transformer.load_weights(MODEL_PATH)
    print("Model and tokenizers loaded successfully.")
    return transformer, eng_tokenizer, fre_tokenizer

def preprocess_sentence(sentence):
    sentence = str(sentence).lower(); sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence); sentence = re.sub(r"[^a-z0-9à-ž?.!,¿]+", " ", sentence, flags=re.IGNORECASE)
    sentence = sentence.strip(); return "sos " + sentence + " eos"

def evaluate(inp_sentence, loaded_transformer, eng_tokenizer, fre_tokenizer):
    inp_sentence_proc = preprocess_sentence(inp_sentence)
    inp_tensor = tf.convert_to_tensor([eng_tokenizer.texts_to_sequences([inp_sentence_proc])[0]])
    inp_tensor = pad_sequences(inp_tensor, maxlen=MAX_LENGTH, padding='post', truncating='post')
    decoder_input = [fre_tokenizer.word_index['sos']]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(MAX_LENGTH):
        predictions, _ = loaded_transformer((inp_tensor, output), training=False)
        predictions = predictions[:, -1:, :]; predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        if predicted_id[0][0] == fre_tokenizer.word_index['eos']: break
    return tf.squeeze(output, axis=0)

def translate(sentence, loaded_transformer, eng_tokenizer, fre_tokenizer):
    result = evaluate(sentence, loaded_transformer, eng_tokenizer, fre_tokenizer)
    predicted_sentence = fre_tokenizer.sequences_to_texts([result.numpy()])
    return predicted_sentence[0].replace('sos ', '').replace(' eos', '').strip()

# ==============================================================================
# FRONTEND: THE STREAMLIT USER INTERFACE
# ==============================================================================

# --- Sidebar ---
with st.sidebar:
    st.title("📖 About the Project")
    st.info(
        "This is a demo application for an English-to-French translation model. "
        "The model is a **Transformer**, a deep learning architecture that has revolutionized "
        "natural language processing."
        "\n\n"
        "The model was trained from scratch on a dataset of over 175,000 sentence pairs."
    )
    st.divider()
    st.header("Technology Stack")
    st.markdown("- **Model:** TensorFlow / Keras")
    st.markdown("- **App:** Streamlit")
    st.markdown("- **Training:** Kaggle Notebook")

# --- Main Page ---
st.title("🤖 AI-Powered French Translator")
st.markdown("This app demonstrates the power of the Transformer architecture for machine translation. Enter an English sentence and see it translated into French in real-time.")

# Create a container for the main content
main_container = st.container(border=True)

with main_container:
    # Get user input
    input_text = st.text_area(
        "**Enter English Text Here:**",
        "The weather is beautiful today.",
        height=100,
        placeholder="e.g., I love to learn new things."
    )

    # Translate button
    if st.button("Translate ➡️", use_container_width=True, type="primary"):
        if input_text:
            # Load assets and perform translation
            try:
                transformer, eng_tokenizer, fre_tokenizer = load_assets()
                with st.spinner("Translating... This may take a moment."):
                    translated_text = translate(input_text, transformer, eng_tokenizer, fre_tokenizer)

                # Display the result in a styled container
                st.divider()
                st.subheader("Translation Result")
                result_container = st.container(border=True)
                with result_container:
                    st.markdown("##### English Input:")
                    st.info(input_text)
                    st.markdown("##### French Translation:")
                    st.success(translated_text)

            except FileNotFoundError:
                st.error(f"**Error:** Model assets not found. Please make sure the '{ASSETS_DIR}' folder is in the same directory as this `app.py` file.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a sentence to translate.")

st.markdown("---")
st.markdown("Created by a dedicated learner, powered by AI.")