from data.data_frame import get_data_frame
def main():
    data_frame = get_data_frame("s_p500")

if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    main()