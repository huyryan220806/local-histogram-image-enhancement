# Do An Cuoi Ky: Tang Cuong Anh Cuc Bo Dua Tren Histogram (Local Image Enhancement Based on Histogram)

Du an nay la do an mon hoc **Nhap Mon Xu Ly Anh So** voi de tai: **"Tang cuong anh cuc bo dua tren histogram"** nham toi muc tieu cai thien chat luong chi tiet cuc bo trong cac hinh anh co do sang hoac do tuong phan khong dong deu.

Du an trien khai, so sanh va danh gia hai thuat toan:
1. **Global Histogram Equalization (HE)**: Can bang histogram toan cuc.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Can bang histogram cuc bo thich ung gioi han do tuong phan (ca phien ban tu cai dat va phien ban OpenCV).

---

## Cau Truc Ma Nguon

| File | Chuc nang |
| :--- | :--- |
| `utils.py` | Cac ham bo tro: Doc/ghi anh Unicode (ho tro tieng Viet), ve histogram, hien thi anh. |
| `global_enhancement.py` | Trien khai can bang histogram toan cuc (Global HE) tu thiet ke & OpenCV tren anh xam va anh mau. |
| `local_enhancement.py` | Trien khai thuat toan CLAHE tu thiet ke & OpenCV tren anh xam va anh mau. Thu nghiem thay doi tham so. |
| `comparison.py` | Tinh toan 5 chi so danh gia (Mean, Std Dev, Entropy, PSNR, SSIM) va ve cac bieu do so sanh. |
| `report_generator.py` | Xuat bao cao tong ket, bang chi so dinh luong, va bieu do so sanh tong hop duoi dang hinh anh va van ban. |
| `main.py` | File khoi chay chinh cua toan bo he thong xu ly anh (chay batch tren thu muc). |
| `app.py` | Giao dien web Streamlit cho phep upload anh va tuong tac truc tiep. |

---

## Giao Dien Web (Streamlit)

Du an co tich hop giao dien web su dung **Streamlit**, cho phep nguoi dung:

- **Upload anh** tu may tinh (JPG, PNG, BMP, TIFF)
- **So sanh truc quan**: Anh goc vs Global HE vs CLAHE canh nhau
- **Dieu chinh tham so**: clipLimit va tileGridSize bang slider
- **Xem Histogram & CDF**: Bieu do phan bo muc xam
- **Xem Metrics**: PSNR, SSIM, Entropy, Std Dev voi nhan xet tu dong
- **Thu nghiem tham so**: Kham pha anh huong cua nhieu bo tham so CLAHE khac nhau
- **Tai xuong**: Luu anh ket qua va bieu do ve may

### Chay giao dien Streamlit

```bash
cd D:\VLU\253\XLAS\DoAnCK
streamlit run app.py
```

Sau do mo trinh duyet tai: **http://localhost:8501**

---

## Huong Dan Cai Dat & Chay

### 1. Yeu cau he thong
* **Python**: Khuyen nghi phien ban 3.10 tro len.
* **Thu vien can cai dat**:
  ```bash
  pip install numpy opencv-python matplotlib streamlit
  ```

### 2. Chay xu ly batch (command line)
1. Dat cac anh dau vao can tang cuong vao thu muc `input/`.
2. Mo terminal tai thu muc du an va chay:
   ```bash
   python main.py
   ```
3. Xem ket qua duoc tao tu dong tai thu muc `output/`.

### 3. Chay giao dien web (Streamlit)
```bash
streamlit run app.py
```

---

## Ket Qua So Sanh Dinh Luong

Duoi day la ket qua do dac trung binh tren bo du lieu anh thu nghiem:

| Chi so (Metric) | Anh Goc | Global HE | CLAHE | Nhan xet |
| :--- | :---: | :---: | :---: | :--- |
| **PSNR (dB)** | - | 11.19 | **35.25** | CLAHE giu do tuong dong cao voi anh goc, tranh meo anh. |
| **SSIM** | 1.0000 | 0.7361 | **0.9906** | SSIM cua CLAHE gan tuyet doi (~0.99) - bao toan cau truc rat tot. |
| **Entropy** | 1.34 | 1.47 | **1.82** | CLAHE tang luong thong tin chi tiet dang ke so voi anh goc & Global HE. |
| **Do lech chuan (Std)** | 35.8 | 87.3 | 44.3 | Global HE lam tang qua muc do tuong phan dan den chay sang. |

---

## Danh Gia Uu & Nhuoc Diem

### 1. Global Histogram Equalization (Toan cuc)
* **Uu diem**: Cuc ky don gian, toc do tinh toan nhanh, khong can tinh chinh tham so dau vao.
* **Nhuoc diem**: 
  * Khong hieu qua doi voi cac anh co do phan bo anh sang khong dong deu.
  * De xay ra hien tuong chay sang o vung sang va mat chi tiet o vung toi.
  * Khuech dai ca cac hat nhieu (noise).

### 2. CLAHE (Cuc bo)
* **Uu diem**:
  * Cai thien chi tiet cuc bo vo cung hieu qua, xu ly xuat sac cac vung bong toi hoac choi sang cuc bo.
  * Gioi han do tuong phan (clipLimit) giup han che toi da viec khuech dai nhieu.
  * Noi suy song tuyen tinh (Bilinear Interpolation) loai bo cac duong bien phan manh khoi (block artifacts).
* **Nhuoc diem**:
  * Thuat toan phuc tap hon, ton nhieu chi phi tinh toan hon.
  * Can lua chon cac tham so phu hop (clipLimit va tileGridSize) de dat hieu qua toi uu cho tung loai anh.

---

## Tac gia
* Ho va ten: **Nguyen Dinh Huy**
* Ma so sinh vien: **2474802010140**
* Truong Dai hoc Van Lang (VLU)
