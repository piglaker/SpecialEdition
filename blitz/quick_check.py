
import sys
sys.path.append("../")
from utils.io import get_sighan_from_json


all_data = get_sighan_from_json()


print(all_data['train'][0])
