if __name__ == "__main__":
    from dashboard import create_dashboard
    from os import system

    system("pip install -r requirements.txt")

    create_dashboard()
