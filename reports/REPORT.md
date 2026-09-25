# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Đỗ Thành Long

Công cụ gán nhãn đã dùng: CVAT Docker chạy trên máy cá nhân

## 1. Dữ liệu và cách chia tập

Camera đứng yên trên cầu vượt và mỗi chiếc xe nằm trong khung hình vài giây. Hai ảnh cách nhau 0.4 giây gần như giống hệt nhau. Vì vậy dữ liệu được chia theo trục thời gian: 20 ảnh kiểm thử lấy từ 4 đoạn có tâm ở giây 20, 60, 100 và 140, 112 ảnh ở giữa bị loại làm vùng đệm, và 268 ảnh còn lại là pool. Ảnh pool gần ảnh kiểm thử nhất vẫn cách 4.4 giây (theo `data/DATA.md`).

Nếu chia ngẫu nhiên, cùng một chiếc xe có thể vừa nằm trong tập học vừa nằm trong tập kiểm thử. Mô hình sẽ được chấm trên những xe nó đã thấy, nên điểm bị lệch theo hướng **cao hơn thực tế**. Đó là rò rỉ dữ liệu (data leakage).

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `reports/rounds_table.md`: `yolov8n cold start (COCO car+bus+truck)`, 0 ảnh train, AP50 **0.771**, P 0.925, R 0.489, F1 0.640. Độ phủ theo kích thước xe: xe nhỏ **0.182**, xe vừa **0.547**, xe lớn **0.561**.

Số này cho thấy xe ở xa bị bỏ sót nhiều nhất. Cứ 5 xe nhỏ thì AI chỉ tìm thấy chưa đến 1. Xe vừa và xe lớn cũng chỉ được tìm thấy khoảng một nửa. Trong `outputs/compare_round0.jpg`, ở `frame_0350` cold start chỉ đúng 9 box (TP9) và sót 14 xe (FN14). Phần lớn xe bị sót là xe nhỏ ở giữa ảnh, chỉ còn cụm đèn. Ở `frame_0250`, một xe sát mép phải cũng bị sót. Cold start còn có 2 khung nhầm (FP2) trên mỗi ảnh, ví dụ khung đỏ quanh cụm đèn hậu ở `frame_0250` và `frame_0350`.

Cần người rà lại nhãn tham chiếu trước khi kết luận AI sai. Nhãn dùng để chấm cũng do một mô hình vẽ và chưa có người xem từng khung (`data/DATA.md`). Ví dụ ở `frame_0150`, cụm đèn hậu đỏ bên phải bị tách thành nhiều khung chồng lên nhau, và ở khung đỏ của `frame_0250` không chắc đó là xe hay ánh phản chiếu. Nếu tham chiếu sai thì đó là lỗi của nhãn chấm, không phải của AI.

## 3. Chiến lược chọn mẫu

Mỗi ảnh trong pool có một điểm `score = W_U·U + W_A·A + W_D·D` với trọng số 0.5, 0.3, 0.2. `U` là mức bất định trung bình của 5 box khó nhất, đạt cao nhất khi độ tin cậy gần 0.5. `A` là số box mơ hồ (conf từ 0.15 đến dưới 0.5), chuẩn hóa theo ảnh nhiều nhất trong pool. `D` là độ xa về thời gian so với ảnh đã có nhãn. `MIN_GAP_S = 2.0` bắt buộc hai ảnh cùng lô cách nhau ít nhất 2 giây, vì camera đứng yên nên ảnh sát nhau gần như là một cảnh.

Ba ảnh đã dẫn trong `SELECTION.md`:
- `frame_0182.jpg` (hạng 1, điểm 0.9591): 28 box trong đó 18 mơ hồ, A = 1.0. Đây là ảnh AI phân vân nhất. Ảnh này cũng có 3 xe bên phải bị gộp chung một khung.
- `frame_0369.jpg` (hạng 2, điểm 0.9324): 43 box, 16 mơ hồ, xe đi rất sát nhau.
- `frame_0099.jpg` (hạng 8, điểm 0.9063): U = 0.946, có xe rất xa trong vùng tối, đúng chỗ AI yếu nhất.

Ảnh thứ tư là `frame_0372.jpg` (hạng 6, điểm 0.9101). Điểm của nó cao hơn `frame_0312` (hạng 7) và `frame_0099` (hạng 8) đã được chọn. AI bỏ qua vì nó cách `frame_0369` chỉ 1.2 giây. Sửa cả hai chỉ tốn công lặp lại cùng một cảnh. Về công gán nhãn, các ảnh dày xe tốn nhiều công hơn hẳn: `frame_0369` phải thêm 24 khung, `frame_0326` thêm 21 khung (`round1_diff.md`).

Điểm bất định **không** chứng minh ảnh đó sẽ cải thiện mô hình. Nó chỉ cho biết AI đang phân vân. AI có thể phân vân vì ảnh khó nhưng học được, hoặc vì ảnh khó và nhãn cũng mơ hồ (xe xa, chỉ còn hai chấm đèn). Ngoài ra lab chỉ chạy một chiến lược, không có chạy `random` để đối chứng, nên không biết cách chọn này có tốt hơn ngẫu nhiên hay không.

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 328 | 0.583 | -0.188 | 1.000 | 0.144 | 0.252 | 0.000 | 0.128 | 0.488 |

**Mức độ sửa nhãn (vòng 1, `outputs/round1_diff.md`).** Trên 12 ảnh, AI đề xuất 169 box. Sau khi sửa còn 328 box: giữ nguyên 145 (accepted), kéo lại 10 (edited), xóa 14 (deleted), thêm mới 173 (added). Tỉ lệ giữ nguyên là 86%. Số box thêm mới lớn cho thấy AI đề xuất thiếu nhiều xe hơn là vẽ sai. Lưu ý khung tách ra từ khung gộp có thể được tính vào `deleted` cộng `added`.

**So sánh điểm.** AP50 vòng 1 là 0.583, **giảm 0.188** so với cold start 0.771. Chưa có vòng 2 nên không có mốc so với vòng trước. Theo kích thước: xe nhỏ từ 0.182 xuống 0.000, xe vừa từ 0.547 xuống 0.128, xe lớn từ 0.561 xuống 0.488. Tất cả nhóm đều xấu đi, xe vừa giảm mạnh nhất. Precision tăng từ 0.925 lên 1.000 (FP về 0), nhưng recall tụt từ 0.489 xuống 0.144 (TP 58, FN 345).

**Một ca đổi (`compare_round0.jpg` và `compare_round1.jpg`).** Ở `frame_0350`, cold start đúng 9 box và sót 14 (TP9 FN14, FP2). Sau fine-tune chỉ đúng 4 box và sót 19 (TP4 FN19, FP0). Khung đỏ nhầm đã biến mất, nhưng nhiều xe từng được tìm thấy ở giữa ảnh nay bị sót. Lý do có thể kiểm: mô hình vòng 1 chỉ vẽ box khi độ tin cậy từ 0.25 trở lên. Precision 1.0 kèm recall 0.14 là dấu hiệu mô hình rất dè dặt, có thể vì chỉ học trên 12 ảnh nên điểm tin cậy còn thấp. Đây là giả thuyết, tôi chưa kiểm chứng. Cách kiểm tra là chạy lại đánh giá ở ngưỡng conf thấp hơn.

**Ba việc khác nhau cần tách riêng.**
- *Quan sát độc lập (`BLIND_SCAN.md`).* Trước khi xem khung AI, tôi nhìn `frame_0099.jpg` và đếm được khoảng 20 xe. Tôi ghi hai chỗ dễ sai: xe ở rìa ảnh chỉ còn đầu xe và ánh đèn, và xe tối màu trong vùng tối hoặc đã khuất, chỉ còn một phần rất nhỏ.
- *Lỗi pre-label đã sửa (`REVIEW_LOG.csv` và `round1_diff.md`).* Ở `frame_0099` AI đề xuất 13 box, tôi giữ 9, kéo 3, xóa 1, thêm 10, còn 22 box. `frame_0182` (xe bên phải bị gộp) và `frame_0270` (dàn xe bên trái bị gộp) được tách thành từng khung riêng.
- *Kết quả mô hình sau train.* AP50 giảm về 0.583 như trên. Đây là số đo trên 20 ảnh kiểm thử, không liên quan trực tiếp đến việc mắt tôi thấy gì hay tôi sửa khung nào.

**Một ca khó theo guideline.** Ở `frame_0182`, ba xe bên phải sát nhau bị AI gộp thành một khung. Theo `GUIDELINE_LABEL.md`, hai xe đứng sát nhau phải có hai box riêng, không gộp. Tôi xóa khung gộp và vẽ khung ôm sát từng xe. Ca này khó vì các xe đi sát và chỉ thấy đèn, phải đoán đường viền thân xe. Ở `frame_0099`, xe rất xa gần như chỉ còn hai chấm đèn, guideline cho phép gán hay không đều được. Tôi chọn vẽ phần thân đoán được quanh cụm đèn, và cố giữ cách này nhất quán.

## 5. Kết luận và giới hạn

So với cold start, vòng 1 **kém hơn**: AP50 từ 0.771 xuống 0.583 (giảm 0.188), recall từ 0.489 xuống 0.144. Tôi chọn **không train thêm ngay**, vì điểm giảm mà chưa rõ lý do. Cho AI học thêm khi chưa hiểu vì sao giảm là lặp lại cùng một lỗi. Tôi cần kiểm tra trước: (1) nhãn vòng 1 có khung sai, khung quá nhỏ hoặc khung trùng không, đặc biệt 173 khung thêm mới; (2) đánh giá lại ở ngưỡng conf thấp hơn để xem mô hình có dè dặt quá không; (3) 50 epoch trên 12 ảnh có làm mô hình học lệch không.

Hai ca còn yếu để làm ở vòng sau:
- *Xe xa, chỉ còn hai chấm đèn* (như `frame_0099`). Recall xe nhỏ về 0.000. Chi phí rà nhãn cao vì ảnh có nhiều xe nhỏ, và guideline không bắt buộc gán nên dễ thiếu nhất quán.
- *Xe đi sát bị gộp khung* (như `frame_0182` và `frame_0270`). Các ảnh dày xe như `frame_0369` tốn nhiều công nhất (thêm 24 khung). Nguy cơ ảnh gần trùng: các ảnh quanh giây 147 đến 149 (`frame_0368`, `0369`, `0372`) gần như một cảnh, nên chỉ nên chọn một ảnh.

Giới hạn ảnh hưởng đến kết luận: tập kiểm thử chỉ có 20 ảnh, và chênh lệch dưới khoảng 0.01 AP50 chưa đủ kết luận. Mức giảm 0.188 lớn hơn nhiều nên khó là do nhiễu, nhưng vẫn chỉ đến từ 4 đoạn video. Các box tham chiếu cao dưới 16 pixel (14 box) bị bỏ qua khi chấm, nên điểm không phản ánh xe rất xa. Nhãn tham chiếu do mô hình tạo và chưa được người rà, nên một phần chênh lệch có thể do nhãn chấm sai chứ không phải AI sai. Ngoài ra chỉ có một lần chạy và chưa có đối chứng `random`, nên tôi không kết luận được cách chọn theo độ bất định tốt hay xấu.
