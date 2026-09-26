# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT — REVIEW 1
## Đề tài: AVIR-KIE (Phân tích bố cục & Trích xuất thông tin hóa đơn/biên lai tiếng Việt)

* **Thời lượng dự kiến:** 15 – 20 phút (bao gồm demo/Q&A).
* **Phân chia người nói:**
  * **Hà Minh Dũng (Leader):** Slides 1, 2, 3.
  * **Nguyễn Phước Đại:** Slides 4, 5.
  * **Nguyễn Khắc Vương:** Slides 6, 7, 8.
  * **Nguyễn Thị Mỹ Cẩm:** Slides 9, 10, 11 và kết thúc.
  * **Cả nhóm:** Slide 12 (Q&A).

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 1: TITLE SLIDE (Giới thiệu đề tài)
###   Người trình bày: Hà Minh Dũng (Leader)
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Dũng]:**
> "Kính chào Thầy Nguyễn Hồng Hải cùng toàn thể các bạn sinh viên và hội đồng có mặt ngày hôm nay. Chúng em là Nhóm 4, hôm nay xin phép được báo cáo tiến độ Review 1 của dự án Capstone Project với đề tài: **AVIR-KIE — Deep Learning-based Layout Analysis and Key Information Extraction from Vietnamese Invoices and Receipts** (Phân tích bố cục và trích xuất thông tin quan trọng từ hóa đơn, biên lai tiếng Việt dựa trên học sâu).
>
> Dự án được hướng dẫn trực tiếp bởi Thầy Nguyễn Hồng Hải. Nhóm chúng em gồm có 4 thành viên: Em là Hà Minh Dũng – Trưởng nhóm, và các bạn thành viên là Nguyễn Phước Đại, Nguyễn Khắc Vương, và Nguyễn Thị Mỹ Cẩm.
>
> Sau đây, em xin phép bắt đầu buổi thuyết trình."
>
> *(Hành động: Click chuyển slide hoặc ra hiệu chuyển slide)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 2: PRESENTATION OUTLINE (Cấu trúc bài thuyết trình)
###   Người trình bày: Hà Minh Dũng (Leader)
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Dũng]:**
> "Để Thầy và hội đồng tiện theo dõi, đây là cấu trúc bài thuyết trình cũng như sự phân công nhiệm vụ của từng thành viên trong buổi hôm nay:
>
> * **Phần của em (Minh Dũng):** Sẽ trình bày giới thiệu chung, bối cảnh bài toán và lý do tại sao chúng em thực hiện đề tài này (Slides 1 - 3).
> * **Bạn Phước Đại:** Sẽ trình bày về Chiến lược dữ liệu (dữ liệu thực tế MC-OCR kết hợp dữ liệu giả lập Synthetic Generator) cùng với sơ đồ cấu trúc luồng Pipeline của hệ thống (Slides 4 - 5).
> * **Bạn Khắc Vương:** Sẽ phụ trách phần Tiền xử lý ảnh bằng OpenCV, mô hình phát hiện chữ CRAFT và mô hình nhận diện chữ VietOCR chuyên dụng cho tiếng Việt (Slides 6 - 8).
> * **Bạn Mỹ Cẩm:** Sẽ đi sâu vào phần trích xuất thông tin thực thể bằng LayoutLMv3, đưa ra ví dụ chạy thử thực tế trong đời thực, giới thiệu Công nghệ sử dụng và Kế hoạch phát triển dự án (Slides 9 - 11).
>
> Sau đây, em xin phép đi vào bối cảnh và lý do chọn đề tài."
>
> *(Hành động: Click sang Slide 3)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 3: CONTEXT & PROBLEM STATEMENT (Bối cảnh & Bài toán)
###   Người trình bày: Hà Minh Dũng (Leader)
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Dũng]:**
> "Trong kỷ nguyên chuyển đổi số hiện nay, các doanh nghiệp và tổ chức hành chính phải đối mặt với một khối lượng khổng lồ các hóa đơn, biên lai, chứng từ tài chính dạng giấy hoặc ảnh chụp mỗi ngày. Việc nhập liệu thủ công (manual entry) vừa tốn thời gian, nhân lực lại vô cùng dễ sai sót.
>
> Một câu hỏi đặt ra là: **Tại sao chúng ta không dùng các hệ thống OCR truyền thống?**
>
> Thực tế, OCR truyền thống chỉ giải quyết được một nửa bài toán: nó đọc ra các chuỗi ký tự dạng thô (Raw Text) mà hoàn toàn **không hiểu được ngữ cảnh cấu trúc** và **mối quan hệ không gian 2D** của tài liệu đó. Ví dụ, nếu OCR đọc được chuỗi số `150.000` trên biên lai, hệ thống không thể tự biết đó là 'Tổng tiền thanh toán' (Total Cost), 'Thuế VAT' (Tax), hay là 'Tiền thối lại' (Change).
>
> Thêm vào đó, hóa đơn biên lai tại Việt Nam cực kỳ phức tạp với hàng trăm loại Layout khác nhau từ các chuỗi siêu thị, nhà thuốc, nhà hàng, cửa hàng tiện lợi... cùng các dấu tiếng Việt phức tạp đòi hỏi bộ từ điển ký tự chuẩn xác.
>
> Do đó, giải pháp của nhóm chúng em là **AVIR-KIE**: Kết hợp xử lý ảnh, phát hiện vùng chữ bằng **CRAFT**, nhận diện tiếng Việt bằng **VietOCR** và cốt lõi là mô hình học sâu đa phương thức **LayoutLMv3** để vừa đọc chữ, vừa phân tích bố cục 2D nhằm tự động phân loại thông tin đầu ra thành cấu trúc JSON chuẩn hóa (Seller, Address, Timestamp, Total Cost, Items).
>
> Tiếp theo, xin mời bạn Phước Đại trình bày về chiến lược dữ liệu và kiến trúc tổng quan."
>
> *(Hành động: Chuyển slide và nhường mic cho Đại)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 4: DATA STRATEGY (Chiến lược dữ liệu)
###   Người trình bày: Nguyễn Phước Đại
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Đại]:**
> "Xin chào Thầy và các bạn, em là Phước Đại. Em xin phép trình bày về **Chiến lược dữ liệu** của dự án.
>
> Đối với các mô hình học sâu, dữ liệu là yếu tố sống còn. Để đạt độ chính xác cao và tránh vi phạm quyền riêng tư thông tin hóa đơn thực tế, chúng em kết hợp hai nguồn dữ liệu:
>
> 1. **Dữ liệu thực tế từ bộ dữ liệu MC-OCR 2021:**
>    * Đây là bộ dữ liệu thực tế gồm khoảng 2.000 ảnh hóa đơn chụp thực tế từ các siêu thị lớn như VinMart, Saigon Co.op...
>    * Dữ liệu này chứa rất nhiều yếu tố nhiễu của đời thực: góc chụp nghiêng, ảnh mờ, hóa đơn bị nhàu nát hoặc thiếu sáng.
>    * Chúng em đã lọc ra các hóa đơn chuẩn theo luồng nhãn để huấn luyện mô hình phát hiện chữ CRAFT và VietOCR.
>
> 2. **Dữ liệu giả lập bằng bộ sinh tự động (Synthetic Generator):**
>    * Chúng em phát triển một script Python sử dụng **Playwright** và công cụ sinh văn bản từ LLM (Gemini API) để tạo ra các hóa đơn tiếng Việt giả lập ngẫu nhiên nhưng đúng ngữ pháp.
>    * Script này kết xuất HTML/CSS thành ảnh PNG/PDF và tự động xuất ra tọa độ hộp bao (Bounding Box) cực kỳ chính xác trực tiếp từ DOM của trình duyệt, đi kèm file nhãn JSON chuẩn cấu trúc của mô hình LayoutLM mà không cần gán nhãn thủ công.
>
> Mục tiêu giai đoạn 1 của chúng em là chuẩn bị tập dữ liệu gồm MC-OCR thực tế kết hợp với khoảng 15 mẫu hóa đơn giả lập để tiến hành gán nhãn đồng thời trên công cụ Label Studio nhằm tạo ra tập dữ liệu tối ưu cho LayoutLMv3."
>
> *(Hành động: Click sang Slide 5)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 5: PIPELINE OVERVIEW (Kiến trúc hệ thống tổng quan)
###   Người trình bày: Nguyễn Phước Đại
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Đại]:**
> "Tiếp theo là **Luồng xử lý (E2E Pipeline)** của hệ thống AVIR-KIE đi qua 4 giai đoạn chính từ ảnh gốc đến kết quả JSON cấu trúc:
>
> * **Giai đoạn đầu vào (Input):** Nhận ảnh chụp hoặc ảnh quét của hóa đơn/biên lai dạng JPG/PNG.
> * **Giai đoạn Tiền xử lý (Preprocessing):** Sử dụng các thuật toán OpenCV để lọc nhiễu, làm rõ nét chữ, và quan trọng nhất là xoay thẳng ảnh nếu ảnh bị nghiêng.
> * **Giai đoạn Phát hiện vùng chữ (Text Detection):** Mô hình **CRAFT** quét ảnh và xác định tọa độ vùng chữ, khoanh vùng các hộp bao (Bounding Boxes).
> * **Giai đoạn Nhận diện chữ (Text Recognition):** Cắt các vùng chữ ra và đưa vào **VietOCR** để chuyển đổi thành văn bản tiếng Việt có dấu chuẩn Unicode.
> * **Giai đoạn Trích xuất thông tin (LayoutLM KIE):** Mô hình **LayoutLMv3** nhận song song 3 luồng thông tin (Từ ngữ, Tọa độ 2D của hộp bao, và Mảnh ảnh đặc trưng) để phân loại nhãn thực thể cho từng hộp chữ.
> * **Đầu ra (Output):** Trả về dữ liệu dạng JSON có cấu trúc gồm các trường thông tin cụ thể để sẵn sàng tích hợp với các phần mềm kế toán ERP hoặc ứng dụng di động.
>
> Sau đây, xin mời bạn Khắc Vương trình bày chi tiết về khâu tiền xử lý ảnh, phát hiện và nhận diện chữ."
>
> *(Hành động: Chuyển slide và nhường mic cho Vương)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 6: IMAGE PREPROCESSING (Tiền xử lý ảnh)
###   Người trình bày: Nguyễn Khắc Vương
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Vương]:**
> "Kính chào Thầy và các bạn, em là Khắc Vương. Em xin phép đại diện nhóm trình bày chi tiết các khâu kỹ thuật bắt đầu từ **Tiền xử lý ảnh bằng OpenCV**.
>
> Ảnh hóa đơn chụp thực tế thường bị lệch, méo, bóng mờ hoặc chất lượng in nhiệt kém. Nếu đưa thẳng vào mô hình OCR, tỷ lệ lỗi sẽ rất cao. Do đó, chúng em xây dựng luồng tiền xử lý gồm 3 bước:
>
> 1. **Deskewing (Xoay thẳng ảnh):** Sử dụng thuật toán biến đổi **Hough Transform** để phát hiện các đường thẳng chủ đạo của hóa đơn, tính toán góc nghiêng và xoay thẳng ảnh về góc 0 độ. Bước này giúp giảm tới 40% lỗi nhận diện dòng chữ bị cắt lệch.
> 2. **Contrast Enhancement (Tăng cường độ tương phản):** Áp dụng thuật toán **CLAHE** (Cân bằng lược đồ xám thích ứng cục bộ) kết hợp binarization tự điều chỉnh bằng phương pháp Otsu. Thuật toán này giúp xử lý các vết đổ bóng, vùng tối sáng không đều do camera điện thoại gây ra.
> 3. **Noise Removal (Khử nhiễu):** Sử dụng bộ lọc Gaussian Blur kết hợp các phép toán hình thái học (Morphological Operations) để loại bỏ các điểm hạt nhiễu nhỏ, làm mịn vùng nền và sắc nét nét chữ.
>
> Kết quả là chúng ta có một bức ảnh phẳng, sắc nét, độ tương phản cao sẵn sàng cho khâu phát hiện chữ."
>
> *(Hành động: Click sang Slide 7)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 7: TEXT DETECTION — CRAFT (Phát hiện vùng chữ)
###   Người trình bày: Nguyễn Khắc Vương
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Vương]:**
> "Sau khi ảnh được làm sạch, chúng em đưa vào mô hình **CRAFT (Character Region Awareness for Text Detection)** để phát hiện chữ.
>
> **Tại sao nhóm chọn CRAFT thay vì các mô hình phát hiện hộp bao truyền thống?**
>
> * Hóa đơn nhiệt có đặc thù là các chữ cái rất nhỏ và khoảng cách dòng cực kỳ khít nhau. CRAFT giải quyết bài toán này bằng cách phát hiện chữ ở **cấp độ ký tự** (Character-level) thay vì quét cả dòng chữ dài ngay từ đầu.
> * Mô hình sử dụng mạng xương sống **VGG-16** để trích xuất đặc trưng đa quy mô, sau đó đưa qua kiến trúc decoder dạng **U-Net** để dự đoán hai bản đồ điểm số quan trọng:
>   * **Region Score Map:** Xác định xác suất một pixel có phải là trung tâm của một ký tự hay không.
>   * **Affinity Score Map:** Xác định mối quan hệ kết nối giữa các ký tự liền kề để ghép chúng lại thành một từ hoặc cụm từ hoàn chỉnh.
> * Từ các bản đồ điểm số này, CRAFT sẽ nhóm các ký tự lại và trả ra danh sách các hộp bao `Bounding Box (x, y, w, h)` của từng từ chính xác, bất kể văn bản có bị cong hay nghiêng nhẹ."
>
> *(Hành động: Click sang Slide 8)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 8: TEXT RECOGNITION — VietOCR (Nhận diện chữ)
###   Người trình bày: Nguyễn Khắc Vương
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Vương]:**
> "Có được danh sách tọa độ hộp bao từ CRAFT, chúng em tiến hành cắt (crop) các vùng ảnh nhỏ chứa từ/dòng chữ đó và đưa vào mô hình **VietOCR** để chuyển từ ảnh sang văn bản chữ viết.
>
> * **VietOCR** là một mô hình thiết kế tối ưu riêng cho tiếng Việt. Bản chất của nó kết hợp cấu trúc **Attention-based Seq2Seq**:
>   * **CNN Backbone (VGG hoặc ResNet):** Nhận diện đặc trưng hình ảnh của dải chữ cắt ra.
>   * **Transformer Encoder:** Học thứ tự ngữ cảnh của chuỗi ký tự.
>   * **Attention Decoder:** Dự đoán từng ký tự đầu ra dựa trên trọng số chú ý, giúp nhận diện chính xác các dấu thanh tiếng Việt (như dấu huyền, sắc, hỏi, ngã, nặng) vốn là điểm yếu của các thư viện OCR quốc tế như Tesseract.
> * Kết quả đầu ra của khâu này là một danh sách các cặp dữ liệu gồm tọa độ hộp bao và chuỗi ký tự văn bản tương ứng (ví dụ: `bbox: [232, 81, 469, 110] -> "VinCommerce"`). Dữ liệu này sẽ được chuyển trực tiếp sang mô hình trích xuất LayoutLMv3 ở Slide tiếp theo.
>
> Sau đây, bạn Mỹ Cẩm sẽ trình bày chi tiết về mô hình LayoutLMv3."
>
> *(Hành động: Chuyển slide và nhường mic cho Cẩm)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 9: KEY INFORMATION EXTRACTION — LayoutLMv3
###   Người trình bày: Nguyễn Thị Mỹ Cẩm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Cẩm]:**
> "Kính chào Thầy và các bạn, em là Mỹ Cẩm. Em xin tiếp tục phần thuyết trình với trái tim của hệ thống: **Mô hình Trích xuất thông tin quan trọng LayoutLMv3**.
>
> **Tại sao chúng em lại chọn LayoutLMv3?**
>
> Như anh Dũng đã nói ở phần đầu, nếu chỉ có chữ thôi thì máy tính không hiểu được. LayoutLMv3 là mô hình học sâu đa phương thức (Multimodal Transformer) tiên phong của Microsoft. Nó không chỉ học ngữ nghĩa từ ngữ (Text) mà còn học cả tọa độ không gian 2D (Bounding Boxes) và đặc trưng hình ảnh của vùng chữ đó (Image Patches). Nó học cách con người đọc tài liệu: chúng ta nhìn thấy một dòng chữ ở góc trên cùng bên trái thì khả năng cao đó là tên cửa hàng (Seller), một dòng số ở cuối cùng kèm chữ 'Tổng' thì đó là tổng tiền (Total Cost).
>
> Trong dự án này, chúng em fine-tune mô hình `microsoft/layoutlmv3-base` để thực hiện tác vụ **Phân loại thực thể chuỗi (Token Classification - NER)** trên 5 lớp nhãn chính:
>
> 1. **SELLER:** Tên thương hiệu/cửa hàng.
> 2. **ADDRESS:** Địa chỉ nơi bán hàng.
> 3. **TIMESTAMP:** Ngày giờ giao dịch in trên hóa đơn.
> 4. **TOTAL_COST:** Tổng số tiền thanh toán cuối cùng của hóa đơn.
> 5. **ITEMS:** Thông tin chi tiết các mặt hàng mua sắm (tên món, số lượng, đơn giá nếu có).
>
> Tập dữ liệu sau khi được gán nhãn trên Label Studio sẽ được định dạng lại tọa độ chuẩn hóa về dải `[0-1000]` để cấp vào mô hình huấn luyện."
>
> *(Hành động: Click sang Slide 10)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 10: REAL-WORLD EXAMPLE (Ví dụ thực tế)
###   Người trình bày: Nguyễn Thị Mỹ Cẩm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Cẩm]:**
> "Để trực quan hóa hoạt động của hệ thống, trên màn hình là một ví dụ thực tế chạy qua Pipeline của chúng em với tệp ảnh hóa đơn siêu thị VinMart thực tế trong tập dữ liệu MC-OCR:
>
> * **Bên trái (① Raw Input Image):** Là ảnh chụp thực tế có mã số `mcocr_public_145013fxcgs.jpg`. Ảnh có độ phân giải `768x1024` pixel, chụp cầm tay bình thường trong điều kiện ánh sáng phòng.
> * **Ở giữa (② CRAFT + VietOCR Output):** Kết quả sau khi chạy qua CRAFT và VietOCR đã phát hiện và trích xuất thành công nội dung của các dòng chữ kèm tọa độ hộp bao của chúng. Ví dụ như dòng chữ `"VinCommerce"` ở tọa độ `[232, 81, 469, 110]` hay số tổng tiền `"30,900"` ở cuối hóa đơn.
> * **Bên phải (③ LayoutLM JSON Output):** Đây là kết quả cuối cùng sau khi LayoutLMv3 xử lý. Các nhãn thực thể đã được gán chính xác vào các trường cấu trúc JSON. Chúng ta thu được thông tin cực kỳ sạch sẽ: `seller` là VinCommerce, `address` là VM+QNH Cam Pha, `timestamp` là ngày 15/08/2020 và `total_cost` là 30,900 VNĐ.
>
> Toàn bộ luồng xử lý này hoạt động khép kín và cho ra kết quả chỉ trong khoảng dưới 2 giây đối với một ảnh hóa đơn thông thường."
>
> *(Hành động: Click sang Slide 11)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 11: TECH STACK & PROJECT TIMELINE (Công nghệ & Lộ trình)
###   Người trình bày: Nguyễn Thị Mỹ Cẩm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Cẩm]:**
> "Về mặt công nghệ và lộ trình thực hiện dự án:
>
> * **Công nghệ sử dụng:**
>   * **AI/ML:** Chúng em dùng **PyTorch** làm framework chủ đạo, kết hợp thư viện **HuggingFace** để huấn luyện LayoutLM, thư viện **OpenCV** để xử lý ảnh và các mô hình pretrained của **CRAFT** và **VietOCR**.
>   * **Hạ tầng & Triển khai:** Dự án dùng **FastAPI** để viết API backend, **Docker** để đóng gói container. Quá trình huấn luyện mô hình sẽ được thực hiện trên GPU Google Colab A100 và triển khai ứng dụng demo trên Hugging Face Spaces.
>
> * **Lộ trình dự án gồm 4 giai đoạn lớn (từ tháng 5 đến tháng 8 năm 2026):**
>   * **Phase 1 (Tháng 5 - Hiện tại):** Đã hoàn thành bộ lọc dữ liệu MC-OCR, viết xong tool tạo hóa đơn giả lập Synthetic Generator và thiết lập công cụ gán nhãn nhóm Label Studio.
>   * **Phase 2 (Tháng 6 - Sắp tới):** Tập trung vào việc tinh chỉnh tham số (fine-tune) đồng thời cả ba mô hình CRAFT, VietOCR và LayoutLMv3 trên GPU để đạt độ chính xác tối ưu.
>   * **Phase 3 (Tháng 7):** Đóng gói API sử dụng FastAPI + Docker và xây dựng giao diện Web Dashboard hiển thị kết quả trực quan cho người dùng cuối.
>   * **Phase 4 (Tháng 8):** Đánh giá độ chính xác (F1-score), viết báo cáo hoàn chỉnh và chuẩn bị bảo vệ đồ án tốt nghiệp.
>
> Dự án cũng lường trước các rủi ro như ảnh méo lệch nặng hoặc thiếu tài nguyên GPU, và nhóm đã chuẩn bị các biện pháp tăng cường dữ liệu cũng như tối ưu hóa kích thước mô hình để giảm thiểu rủi ro này."
>
> *(Hành động: Click sang Slide 12)*

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 12: Q&A (Hỏi & Đáp)
###   Người trình bày: Cả nhóm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại kết thúc của Cẩm]:**
> "Dạ, phần trình bày về tiến độ báo cáo Review 1 đề tài **AVIR-KIE** của nhóm chúng em đến đây là kết thúc. Chúng em xin cảm ơn Thầy Nguyễn Hồng Hải cùng hội đồng đã chú ý lắng nghe.
>
> Nhóm chúng em rất mong nhận được những ý kiến đóng góp, nhận xét và câu hỏi của Thầy cùng các bạn để hoàn thiện dự án tốt hơn trong các giai đoạn tiếp theo.
>
> Chúng em xin chân thành cảm ơn!"
>
> *(Hành động: Tất cả thành viên cúi đầu chào và chuẩn bị trả lời câu hỏi phản biện)*
