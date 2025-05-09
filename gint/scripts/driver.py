from gint.host.cuda.driver import DriverContext, read_ptx, ptx_link, launch_kernel


def main():
    with DriverContext(0) as driver_ctx:
        context = driver_ctx.context
        print(f"Created context: {context}")

        ptx_file = "test.ptx"
        kernel_name = b"geval"
        ptx_content = read_ptx(ptx_file)

        function = ptx_link(driver_ctx, ptx_content, kernel_name, verbose=2)

        launch_kernel(
            function, 42,
            grid_dim=2,
            block_dim=64,
            sync=True
        )


if __name__ == "__main__":
    main()
