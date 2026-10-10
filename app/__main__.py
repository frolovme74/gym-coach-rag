import argparse

from .server import create_server


def main():
    parser = argparse.ArgumentParser(prog="python -m app", description="Gym Coach RAG")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"Gym Coach запущен: http://{args.host}:{args.port}  (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
