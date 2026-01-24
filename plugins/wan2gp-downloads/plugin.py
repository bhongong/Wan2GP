"""Downloads plugin for WAN2GP - provides UI for downloading in-house LoRA collections."""

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download
from typing import Iterator, Tuple
import logging

# Set up logger for this plugin
logger = logging.getLogger(__name__)


class DownloadsPlugin(WAN2GPPlugin):
    """Plugin that provides a downloads tab for fetching official LoRA collections."""

    def __init__(self) -> None:
        """Initialize the Downloads plugin with metadata."""
        super().__init__()
        self.name = "Downloads Tab"
        self.version = "1.0.0"
        self.description = "Download our in-house loras!"

    def setup_ui(self) -> None:
        """Register global functions, components, and create the downloads tab."""
        self.request_global("get_lora_dir")
        self.request_global("refresh_lora_list")

        self.request_component("state")
        self.request_component("lset_name")
        self.request_component("loras_choices")

        self.add_tab(
            tab_id="downloads",
            label="Downloads",
            component_constructor=self.create_downloads_ui,
        )

    def create_downloads_ui(self) -> None:
        """Build the UI layout for the downloads tab."""
        with gr.Row():
            with gr.Row(scale=2):
                gr.Markdown(
                    "<I>WanGP's Lora Festival! Press the following button to download i2v "
                    "<B>Remade_AI</B> Loras collection (and bonus Loras).</I>"
                )
            with gr.Row(scale=1):
                self.download_loras_btn = gr.Button(
                    "---> Let the Lora's Festival Start!", scale=1
                )
            with gr.Row(scale=1):
                gr.Markdown("")
        self.download_status = gr.Markdown()
        self.download_loras_btn.click(
            fn=self.download_loras_action,
            inputs=[],
            outputs=[self.download_status],
        ).then(
            fn=self.refresh_lora_list,
            inputs=[self.state, self.lset_name, self.loras_choices],
            outputs=[self.lset_name, self.loras_choices],
        )

    def download_loras_action(self) -> Iterator[str]:
        """Download LoRAs from the Hugging Face repo if not already downloaded.

        Returns:
            Iterator[str]: Status messages (HTML) streamed to the UI.
        """
        yield "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>"
        
        lora_dir = self.get_lora_dir("i2v")
        log_path = os.path.join(lora_dir, "log.txt")
        
        if os.path.isfile(log_path):
            logger.info("LoRAs already downloaded, skipping.")
            yield (
                "<B><FONT SIZE=3>Lora's Festival is already ON! "
                "(Loras already downloaded)</B></FONT>"
            )
            return

        # Download LoRAs into a temporary directory
        tmp_path = os.path.join(lora_dir, "tmp_lora_download")
        logger.info(f"Downloading LoRAs to temporary path: {tmp_path}")
        
        try:
            snapshot_download(
                repo_id="DeepBeepMeep/Wan2.1",
                allow_patterns="loras_i2v/*",
                local_dir=tmp_path,
            )
        except Exception as e:
            logger.error(f"Failed to download LoRAs: {e}")
            yield f"<B><FONT SIZE=3 COLOR=red>Download failed: {e}</B></FONT>"
            return

        # Move downloaded files to the main LoRA directory
        loras_subdir = os.path.join(tmp_path, "loras_i2v")
        for f in glob.glob(os.path.join(loras_subdir, "*")):
            if os.path.isfile(f):
                target_file = os.path.join(lora_dir, os.path.basename(f))
                if os.path.exists(target_file):
                    logger.info(f"Removing old file: {target_file}")
                    os.remove(target_file)
                logger.info(f"Moving {f} to {lora_dir}")
                shutil.move(f, lora_dir)

        # Clean up temporary directory
        try:
            shutil.rmtree(tmp_path)
            logger.info(f"Removed temporary directory: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to remove tmp_path: {e}")

        # Write log file with timestamp
        dt = datetime.today().strftime("%Y-%m-%d")
        tm = datetime.now().strftime("%H:%M:%S")
        try:
            with open(log_path, "w", encoding="utf-8") as writer:
                writer.write(f"Loras downloaded on {dt} at {tm}")
            logger.info(f"LoRAs successfully downloaded on {dt} at {tm}")
        except Exception as e:
            logger.error(f"Failed to write log file: {e}")

        yield "<B><FONT SIZE=3 COLOR=green>Lora's Festival is successfully STARTED!</B></FONT>"
