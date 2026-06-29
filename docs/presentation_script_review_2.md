# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT — REVIEW 2
## Đề tài: AVIR-KIE (Phân tích bố cục & Trích xuất thông tin hóa đơn/biên lai tiếng Việt)

* **Thời lượng dự kiến:** 15 – 20 phút (bao gồm Q&A và phản biện).
* **Mục tiêu cốt lõi:** Bảo vệ hướng đi của nhóm, đập tan hiểu lầm của Thầy Huy về LLM/RAG, và báo cáo kết quả Data + Baseline.
* **Phân chia người nói:**
  * **Hà Minh Dũng (Leader):** Slides 1, 2, 3 (Mở đầu, Đập tan hiểu lầm của thầy, Hướng đi Paper).
  * **Nguyễn Phước Đại:** Slides 4, 5 (Data & Pipeline).
  * **Nguyễn Khắc Vương:** Slides 6, 7 (Baseline Models & Metrics).
  * **Nguyễn Thị Mỹ Cẩm:** Slides 8, 9 (Tiến độ, Định hướng & Kết thúc).

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 1 & 2: GIỚI THIỆU & GIẢI ĐÁP PHẢN BIỆN LẦN 1
###   Người trình bày: Hà Minh Dũng (Leader)
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Dũng]:**
> "Kính chào Thầy Nguyễn Hồng Hải, Thầy Huy cùng toàn thể hội đồng. Chúng em là Nhóm 4, hôm nay xin phép được báo cáo tiến độ Review 2 của dự án **AVIR-KIE — Trích xuất thông tin hóa đơn tiếng Việt**. 
>
> Trước khi đi vào chi tiết kỹ thuật, em xin phép dành slide đầu tiên này để **trực tiếp giải đáp những câu hỏi đóng góp của Thầy Huy trong buổi Review 1**, nhằm làm rõ hoàn toàn bản chất bài toán mà nhóm đang giải quyết.
>
> Ở lần báo cáo trước, có vẻ hội đồng đang hiểu lầm rằng nhóm sử dụng các Mô hình Ngôn ngữ Lớn (LLM) như ChatGPT hay LLaMA kết hợp với kỹ thuật RAG. Em xin khẳng định lại: **Dự án của chúng em KHÔNG sử dụng LLM và cũng KHÔNG tốn bất kỳ chi phí API nào ra bên ngoài.** 
>
> Cụ thể, trả lời câu hỏi của thầy Huy về *'Lưu data input/output ở đâu và chunking tính toán token như thế nào?'*: Vì không dùng LLM nên nhóm không sử dụng Vector Database hay cơ chế Chunking sinh văn bản. Bài toán của nhóm là **Token Classification (Phân loại thực thể chuỗi)** sử dụng họ mô hình Transformer Encoder. Input đưa vào là một bức ảnh và đầu ra là một file JSON chứa các nhãn dán cụ thể. Toàn bộ quá trình từ OCR đến trích xuất thông tin đều được chạy cục bộ (Local/Offline) hoàn toàn 100% bằng mã nguồn mở."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 3: ĐỊNH HƯỚNG BÁO CÁO KHOA HỌC (FISAT)
###   Người trình bày: Hà Minh Dũng (Leader)
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Dũng]:**
> "Và cũng để trả lời câu hỏi của thầy về *'3 câu hỏi nghiên cứu và phương án ra Paper'* – Nhóm chúng em đặt mục tiêu sẽ mang công trình này nộp tại Hội nghị FISAT sắp tới, với 3 câu hỏi nghiên cứu (Research Questions) cực kỳ rõ ràng:
>
> **Thứ nhất**, việc kết hợp thông tin thị giác (Visual) và không gian (Layout) giúp cải thiện độ chính xác bao nhiêu phần trăm so với các mô hình chỉ dùng thuần văn bản (Text-only) trên hóa đơn tiếng Việt?
> **Thứ hai**, phương pháp tổng hợp dữ liệu nhân tạo (Synthetic Data) giải quyết sự thiếu hụt dữ liệu gán nhãn như thế nào?
> **Thứ ba**, các công cụ OCR khác nhau tác động ra sao đến hiệu năng cuối cùng của KIE?
>
> Để trả lời các câu hỏi này, nhóm đã phải xây dựng một bộ dữ liệu khổng lồ. Và sau đây, em xin nhường lời cho bạn Phước Đại trình bày về chiến công lớn nhất của nhóm trong tháng qua: Bộ Dữ Liệu."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 4: DỮ LIỆU & EDA
###   Người trình bày: Nguyễn Phước Đại
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Đại]:**
> "Xin chào Thầy và hội đồng, em là Phước Đại. Để huấn luyện một mô hình Deep Learning thì dữ liệu là linh hồn. Chúng em tự hào trình bày về bộ Dataset nội bộ mà nhóm đã dày công xây dựng:
>
> * Về **Dữ liệu thực tế**, bên cạnh việc chia nhau gán nhãn chi tiết hàng trăm ảnh từ tập MCOCR 2021, nhóm em đã tự bỏ tiền túi đi in ấn và chụp tay **hơn 2.000 hóa đơn/biên lai thực tế** thu thập từ khắp các cửa hàng, siêu thị hiện nay. Điều này giúp bổ sung các mẫu layout hóa đơn mới nhất của năm 2024 mà tập MCOCR cũ không có.
>
> * Về **Dữ liệu giả lập (Synthetic)**, chúng em đã dùng Python trộn template với từ điển thông tin giả để tự động sinh ra thêm **10.000 ảnh hóa đơn**. Ưu điểm tuyệt đối của phương pháp này là 10.000 ảnh này sinh ra đã CÓ SẴN tọa độ Bounding Box và Nhãn chính xác 100%, chúng em không tốn một giây nào để gán nhãn tay cho chúng!
>
> Phân tích dữ liệu (EDA) cho thấy có sự mất cân bằng nhãn rất lớn (nhãn TOTAL_COST xuất hiện rất ít so với nhãn O - Other). Chúng em đã thiết kế Loss Function để khắc phục nhược điểm này. Mời hội đồng xem tiếp Pipeline tổng thể ở Slide 5."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 5: PIPELINE END-TO-END RÕ RÀNG
###   Người trình bày: Nguyễn Phước Đại
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Đại]:**
> "Để giải quyết góp ý của Thầy Huy về việc *'Pipeline chưa rõ ràng'*, chúng em đã vẽ lại Sơ đồ Luồng xử lý một cách tuyến tính và tường minh nhất. Nó hoạt động tuần tự như sau:
>
> Đầu tiên, hệ thống nhận **Ảnh Hóa Đơn**. Ảnh này đi qua khâu **Tiền xử lý** để xoay thẳng, cắt viền. Tiếp theo, nó đi qua **OCR Engine** (PaddleOCR hoặc CRAFT) để trích xuất ra tọa độ (Bounding Boxes) và chữ (Text). Cuối cùng, Tọa độ và Chữ này được ném vào **Mô hình Trích xuất (KIE Model)** để dự đoán xem chữ đó thuộc nhãn gì (Tên quán, Địa chỉ hay Tổng tiền). Cuối cùng trả ra JSON. Cực kỳ gọn gàng và khép kín!
>
> Về việc so sánh các Baseline Model trong khâu KIE, xin mời bạn Khắc Vương trình bày."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 6: SO SÁNH 3 BASELINE & KIẾN TRÚC LAYOUTLMV3
###   Người trình bày: Nguyễn Khắc Vương
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Vương]:**
> "Kính chào Thầy, em là Khắc Vương. Trả lời cho yêu cầu *'So sánh 3 mô hình Baseline tương đương tham số'* của Thầy Huy, nhóm đã chọn ra 3 đại diện tiêu biểu nhất có cùng cân nặng khoảng **110 đến 130 triệu tham số**:
>
> 1. **PhoBERT-base:** Chỉ học chữ. Hiểu tiếng Việt rất giỏi nhưng bị 'mù' vị trí (không biết chữ nào nằm trên hay dưới).
> 2. **LayoutLMv1:** Học chữ + Tọa độ 2D. Hiểu được cấu trúc nhưng lại chưa nhìn thấy 'màu sắc' hay 'đậm nhạt' của ảnh.
> 3. Và **LayoutLMv3:** Đây cũng là câu trả lời cho thắc mắc *'LayoutLMv3 đặc biệt ở chỗ nào?'*. Khác với các bản trước, LayoutLMv3 mã hóa **CÙNG LÚC 3 thứ**: Text, Tọa độ, và **Hình ảnh (Image Patches)**. Nhờ kỹ thuật Linear Projection giống như ViT, mô hình trực tiếp 'nhìn' vào nét đậm nhạt, logo trên ảnh để phán đoán. Đó là lý do nó vượt trội hoàn toàn.
>
> Sang slide tiếp theo, em xin nói về cách nhóm đánh giá độ chính xác."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 7: METRICS (ĐỘ ĐO ĐÁNH GIÁ)
###   Người trình bày: Nguyễn Khắc Vương
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Vương]:**
> "Về độ đo đánh giá (Metrics), vì đây là bài toán nối tiếp từ OCR, nên nếu OCR nhận diện sai một dấu ngã (ví dụ 'Thành tiền' thành 'Thành tiên') thì việc đánh giá Exact Match (Khớp tuyệt đối) sẽ làm kết quả cực kỳ thấp và bất công cho mô hình KIE.
>
> Do đó, nhóm sử dụng độ đo **Fuzzy F1-Score (Khớp tương đối)** dựa trên khoảng cách Levenshtein. Chỉ cần chuỗi trích xuất giống khoảng 80% so với nhãn gốc thì sẽ được tính là đoán đúng (True Positive). Cách đánh giá này sát với thực tế môi trường doanh nghiệp hiện nay.
>
> Phần cuối cùng về kế hoạch thực hiện, xin mời bạn Mỹ Cẩm."

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 8: TIẾN ĐỘ & ĐỊNH HƯỚNG
###   Người trình bày: Nguyễn Thị Mỹ Cẩm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Cẩm]:**
> "Kính chào Thầy và hội đồng, em là Mỹ Cẩm. Để tổng kết lại buổi Review 2, em xin cập nhật tiến độ hiện tại.
>
> Nhóm đã hoàn thiện xong toàn bộ Pipeline, thu thập và sinh thành công tập dữ liệu khổng lồ (2.000 ảnh thật + 10.000 ảnh ảo). Hiện tại, chúng em đang cắm máy chạy Benchmark và Huấn luyện (Fine-tune) các mô hình trên Kaggle.
>
> **Kế hoạch sắp tới của nhóm:**
> Thứ nhất, đóng gói model thành một Demo hoàn chỉnh sử dụng FastAPI và Web Dashboard. 
> Thứ hai, tối ưu hóa tốc độ OCR để hệ thống có thể chạy Real-time.
> Và quan trọng nhất là tổng hợp các chỉ số Benchmark, hoàn thiện bản thảo nghiên cứu (Paper) để nộp tham dự Hội nghị FISAT đúng hạn.
>
> *(Chuyển sang Slide 9 - Q&A)*"

---

### ╔════════════════════════════════════════════════════════════╗
###   SLIDE 9: KẾT THÚC & Q&A
###   Người trình bày: Nguyễn Thị Mỹ Cẩm & Cả nhóm
### ╚════════════════════════════════════════════════════════════╝

**[Lời thoại của Cẩm]:**
> "Dạ, phần trình bày Review 2 của nhóm chúng em đến đây là kết thúc. Chúng em xin cảm ơn Thầy cùng hội đồng đã chú ý lắng nghe sự tiến bộ của nhóm so với Review 1.
>
> Chúng em rất sẵn lòng giải đáp mọi thắc mắc hoặc câu hỏi phản biện từ phía Thầy Cô ạ. Em xin cảm ơn!"
>
> *(Hành động: Toàn bộ nhóm sẵn sàng bật mic/bước lên phía trước để Q&A)*
