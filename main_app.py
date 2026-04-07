import logging

from pipeline import MonitorPipeline

# Configure logging for the entire project
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", mode="a")],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Application...")
    try:
        pipeline = MonitorPipeline()
        temperature, resp = pipeline.run()
        return temperature, resp
    except Exception as e:
        import traceback

        logger.error(f"Application encountered a fatal error: {e}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        temperature, Resp = result
        logger.info(f"Final session temperature : {round(temperature, 2)}")
        logger.info(f"Final session BPM : {round(Resp, 2)}")
    else:
        logger.error("Application exited with an error.")
