import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- 頁面設定 ---
st.set_page_config(page_title="HK GBP Auto-Checker", layout="wide")

st.title("🏗️ 香港建築圖則自動檢查系統 (GBP Auto-Checker)")
st.markdown("**針對香港 GBP / A&A 圖則設計的自動面積計算 MVP**")

# --- 側邊欄：控制台 ---
st.sidebar.header("1. 圖則輸入")
uploaded_file = st.sidebar.file_uploader("上傳平面圖 (JPG/PNG)", type=["jpg", "png", "jpeg"])

st.sidebar.header("2. 參數設定")
# 這是給測量師調整的「魔術棒靈敏度」
threshold_val = st.sidebar.slider("二值化閾值 (Threshold)", 0, 255, 200, help="越低越黑，越高越白。用來過濾淡色的雜訊。")
blur_kernel = st.sidebar.slider("雜訊模糊 (Blur)", 1, 15, 5, step=2, help="數值越大，越能把斷開的牆連起來，但也可能糊掉窗戶。")
min_area_filter = st.sidebar.number_input("最小房間面積過濾 (m²)", value=2.0, help="小於這個面積的區域會被視為雜訊（如柱子、廁所管道）。")

st.sidebar.header("3. 比例尺校正")
# 在 Web 版還沒做互動畫線前，先用數值代替
scale_ratio = st.sidebar.number_input("比例尺 (像素/米)", value=50.0, help="例如：圖上 50px 代表現實 1米")

# --- 核心處理函數 ---
def process_image(pil_image, thresh_val, blur_val, scale, min_area_m2, highlight_id=None):
    # PIL 轉 OpenCV 格式
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR
    original = img_cv.copy()
    
    # 獲取圖片尺寸，用於計算字體大小
    h, w = original.shape[:2]
    # 經驗公式：圖片越寬，字越大。例如寬 2000px 時，字體大小約 2.0
    dynamic_font_scale = w / 1000.0 
    dynamic_thickness = max(2, int(w / 500.0))

    # 1. 轉灰階
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 圖像預處理
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 形態學操作
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # 4. 找輪廓
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms_data = []
    output_img = original.copy()
    
    # 如果有指定高亮房間，先將整張圖變暗 (Dimming)
    if highlight_id and highlight_id != "顯示全部 (Show All)":
        output_img = cv2.addWeighted(output_img, 0.3, np.zeros_like(output_img), 0.7, 0)

    room_id = 1
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if scale > 0:
            area_m2 = area_px / (scale ** 2)
        else:
            area_m2 = 0
            
        if area_m2 > min_area_m2:
            current_id = f"R{room_id}"
            
            # --- 決定顏色與繪圖邏輯 ---
            is_highlighted = (highlight_id == current_id)
            show_all = (highlight_id is None or highlight_id == "顯示全部 (Show All)")
            
            # 只有在「顯示全部」或「選中該房間」時才畫
            if show_all or is_highlighted:
                
                # 顏色：選中用紅色，普通用綠色
                color = (0, 0, 255) if is_highlighted else (0, 255, 0)
                thickness = dynamic_thickness * 2 if is_highlighted else dynamic_thickness
                
                # 畫框
                cv2.drawContours(output_img, [cnt], -1, color, thickness)
                
                # 標註中心點
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # 標籤文字
                    label = current_id
                    if is_highlighted:
                        label += f" ({area_m2:.1f}m2)" # 高亮時顯示面積
                    
                    # --- 繪製帶背景的文字 (讓字看得清) ---
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, dynamic_thickness)
                    
                    # 畫白色背景框
                    cv2.rectangle(output_img, (cX, cY - text_h - 10), (cX + text_w, cY + 5), (255, 255, 255), -1)
                    # 畫文字
                    cv2.putText(output_img, label, (cX, cY), 
                               cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, color, dynamic_thickness)

            rooms_data.append({
                "Room ID": current_id,
                "Area (m²)": round(area_m2, 3),
            })
            room_id += 1
            
    return output_img, rooms_data, processed
# --- 主界面邏輯 ---

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始圖則")
        st.image(image, use_column_width=True)
        
     # ... (在讀取 image = Image.open(uploaded_file) 之後) ...

    # --- 1. 預先執行一次分析以獲取房間列表 (為了做 Dropdown) ---
    # 這裡我們只跑一次計算，不畫圖，純粹為了拿 Data
    # 注意：為了效能，這不是最優解，但在 MVP 裡這樣寫最簡單
    _, initial_data, _ = process_image(image, threshold_val, blur_kernel, scale_ratio, min_area_filter)
    
    # 提取所有房間 ID
    room_options = ["顯示全部 (Show All)"]
    if initial_data:
        room_list = [item["Room ID"] for item in initial_data]
        room_options += room_list

    # --- 2. 搜尋/高亮功能 ---
    st.markdown("### 🔍 房間定位器 (Room Locator)")
    selected_room = st.selectbox("選擇你想查看的房間 ID：", room_options)

    # --- 3. 再次執行分析 (帶著高亮參數) ---
    result_img, data, debug_img = process_image(image, threshold_val, blur_kernel, scale_ratio, min_area_filter, highlight_id=selected_room)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始圖則")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI 識別結果")
        # 顯示結果圖
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
      

    # 顯示數據表格
    st.subheader("📊 智能面積表 (Interactive GFA Schedule)")

    if data:
        # 1. 準備數據框架 (DataFrame)
        df = pd.DataFrame(data)
        
        # 2. 加入互動功能：
        #    - "Exempt?": 讓用戶勾選是否豁免 (預設 False)
        #    - "Room Name": 讓用戶修改房間名稱 (預設是用 ID)
        #    - "Remarks": 備註欄
        
        # 為了避免每次刷新都重置，我們檢查是否已經有數據，或者初始化
        if "Exempt?" not in df.columns:
            df.insert(0, "Exempt? (豁免)", False)  # Checkbox
        if "Room Name" not in df.columns:
            df.insert(1, "Room Name", df["Room ID"]) # Text Input
        df["Remarks"] = "" # Text Input

        # 3. 顯示可編輯表格 (Data Editor)
        st.info("💡 **AP 小貼士**：你可以直接在表格中修改「房間名稱」或勾選「豁免」(如機房/露台)，總 GFA 會自動扣除！")
        
        # 這是 Streamlit 的核心互動元件
        edited_df = st.data_editor(
            df,
            column_config={
                "Exempt? (豁免)": st.column_config.CheckboxColumn(
                    "豁免 GFA?",
                    help="勾選後，此房間將不會計入總 GFA (例如: Plant Room, Refuge Floor)",
                    default=False,
                ),
                "Room Name": st.column_config.TextColumn(
                    "房間名稱",
                    help="點擊修改，例如改為 'Master Bedroom'",
                    required=True,
                ),
                "Area (m²)": st.column_config.NumberColumn(
                    "面積 (m²)",
                    format="%.3f",
                    disabled=True, # 面積由 AI 算，不讓改
                ),
                "Remarks": st.column_config.TextColumn(
                    "備註",
                    help="例如: B(P)R 23(3)(b) Exemption",
                ),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic" # 允許用戶刪除錯誤識別的行
        )

        # 4. 動態計算 GFA (只計算未勾選豁免的項目)
        # 邏輯：只選取 [Exempt? (豁免)] == False 的行
        gfa_df = edited_df[edited_df["Exempt? (豁免)"] == False]
        exempt_df = edited_df[edited_df["Exempt? (豁免)"] == True]

        total_area = edited_df["Area (m²)"].sum()
        total_gfa = gfa_df["Area (m²)"].sum()
        exempted_area = exempt_df["Area (m²)"].sum()

        # 5. 顯示結果看板 (Dashboard Metrics)
        st.write("---")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(label="總建築面積 (Total Construction Area)", value=f"{total_area:.3f} m²")
        with m2:
            st.metric(label="❌ 豁免面積 (Exempted / Disregarded)", value=f"{exempted_area:.3f} m²", delta=f"-{exempted_area:.3f}")
        with m3:
            st.metric(label="✅ **總樓面面積 (Total GFA)**", value=f"**{total_gfa:.3f} m²**", delta_color="normal")

        # 6. 下載最終報告
        # 準備 Excel 下載 (這裡存成 CSV)
        csv = edited_df.to_csv(index=False).encode('utf-8-sig') # 使用 sig 編碼解決 Excel 中文亂碼
        
        st.download_button(
            label="📥 下載 GFA 計算表 (CSV)",
            data=csv,
            file_name="GFA_Schedule_Calculated.csv",
            mime="text/csv",
        )

    else:
        st.warning("⚠️ 找不到符合條件的房間，請嘗試調整左側的 '閾值' 或 'Blur' 參數。")

    # Debug 模式 (保持不變)
    with st.expander("🔧 開發者視圖 (Debug View - 二值化影像)"):
        st.write("這是電腦真正看到的畫面（只有黑白）。如果這裡斷開了，面積就算不到。")
        if 'debug_img' in locals():
            st.image(debug_img, use_column_width=True, channels="GRAY")
