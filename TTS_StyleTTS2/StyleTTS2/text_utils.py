# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

# class TextCleaner:
#     def __init__(self, dummy=None):
#         self.word_index_dictionary = dicts
#         print(len(dicts))
#     def __call__(self, text):
#         indexes = []
#         for char in text:
#             try:
#                 indexes.append(self.word_index_dictionary[char])
#             except KeyError:
#                 print(text)
#         return indexes

# CODE MỚI SO VỚI REPO GỐC
# SỬA LỖI: Thay thế TextCleaner cũ bằng TextCleaner mới, sử dụng phoneme_vocab.json để ánh xạ ký tự thành ID, và gán <unk> cho các ký tự không có trong từ điển.
# Lưu ý: Bạn cần đảm bảo rằng đường dẫn đến phoneme_vocab.json là chính xác và file này tồn tại, nếu không sẽ gây lỗi khi khởi tạo TextCleaner.
# Bạn cũng cần đảm bảo rằng phoneme_vocab.json có cấu trúc đúng với "char_to_id" chứa ánh xạ ký tự sang ID.
import json

class TextCleaner:
    def __init__(self, dummy=None):
        # Trỏ đường dẫn tuyệt đối đến phoneme_vocab.json của bạn
        vocab_path = "D:/HUST_Project/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output/phoneme_vocab.json"
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            
        self.char_to_id = vocab_data["char_to_id"]
        self.unk_id = self.char_to_id.get("<unk>", 1)

    def __call__(self, text):
        # Đảm bảo KHÔNG có lệnh print(text) nào ở đây
        sequence = []
        for char in text:
            # Chuyển đổi từng ký tự thành ID, nếu không có trong từ điển thì gán bằng <unk>
            sequence.append(self.char_to_id.get(char, self.unk_id))
        return sequence
