
import streamlit as st
import numpy as np 
import pandas as pd

# Đọc dữ liệu từ tệp .pkl

st.header("Books Recommender System using Machine Learning")
model =  pd.read_pickle('artifact/model.pkl')
books_name =  pd.read_pickle('artifact/books_name.pkl')
final_rating = pd.read_pickle('artifact/final_rating.pkl')
book_pivot  = pd.read_pickle("artifact/book_pivot.pkl")
top_10_book_recommend  = pd.read_pickle("artifact/top_10_book_recommend.pkl")


# Thêm một chuỗi rỗng vào đầu danh sách books_name
books_name = pd.Index(['Choose a book'] + list(books_name))

def fecth_poster(suggestion):
    book_name= []
    ids_index = []
    poster_url= []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title']== name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        poster_url.append(url)
    
    return poster_url



def recommend_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=11)

    poster_url = fecth_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list, poster_url


import streamlit as st

# Khởi tạo session state cho selected_books nếu chưa tồn tại và giá trị là rỗng
if 'selected_books' not in st.session_state:
    st.session_state.selected_books = ""

# Hiển thị selectbox
selected_books = st.selectbox(
    "Type or select a book",
    [""] + books_name
)

# Cập nhật giá trị session state khi selected_books thay đổi
st.session_state.selected_books = selected_books



if selected_books == "Choose a book":
    st.info("Top 10 most highly rated books:")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Thiết lập chiều cao cho mỗi cột
    for col in [col1, col2, col3, col4, col5]:
        col.write(
            """
            <style>
            .book-container {
                height: 300px;
                margin-right: 10px;
            }
            .image {
                margin-top: 20px;
                height: 220px;
                width: 130px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    books_iter = iter(top_10_book_recommend.iterrows())
    for index, row in books_iter:
        with col1:
            st.markdown(f"""<div class="book-container"><img class="image" src='{row['img_url']}'><p>{row['title']}</p></div>""", unsafe_allow_html=True)
        try:
            with col2:
                row = next(books_iter)[1]
                st.markdown(f"""<div class="book-container"><img class="image"  src='{row['img_url']}'><p>{row['title']}</p></div>""", unsafe_allow_html=True)
            with col3:
                row = next(books_iter)[1]
                st.markdown(f"""<div class="book-container"><img class="image"  src='{row['img_url']}'><p>{row['title']}</p></div>""", unsafe_allow_html=True)
            with col4:
                row = next(books_iter)[1]
                st.markdown(f"""<div class="book-container"><img class="image" src='{row['img_url']}'><p>{row['title']}</p></div>""", unsafe_allow_html=True)
            with col5:
                row = next(books_iter)[1]
                st.markdown(f"""<div class="book-container"><img class="image"  src='{row['img_url']}'><p>{row['title']}</p></div>""", unsafe_allow_html=True)
        except StopIteration:
            break



else:
    if st.button('Show recommendation'):
        recommendation_books, poster_url = recommend_books(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)
        for col in [col1, col2, col3, col4, col5]:
            col.write(
                """
                <style>
                .book-container {
                    height: 300px;
                    margin-right: 10px;
                }
                .image {
                    margin-top: 20px;
                    height: 220px;
                    width: 130px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        with col1:
            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[1]}" class="image">
                    <p>{recommendation_books[1]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[6]}" class="image">
                    <p>{recommendation_books[6]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[2]}" class="image">
                    <p>{recommendation_books[2]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[7]}" class="image">
                    <p>{recommendation_books[7]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[3]}" class="image">
                    <p>{recommendation_books[3]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[8]}" class="image">
                    <p>{recommendation_books[8]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[4]}" class="image">
                    <p>{recommendation_books[4]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[9]}" class="image">
                    <p>{recommendation_books[9]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col5:
            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[5]}" class="image">
                    <p>{recommendation_books[5]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="book-container">
                    <img src="{poster_url[10]}" class="image">
                    <p>{recommendation_books[10]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )





