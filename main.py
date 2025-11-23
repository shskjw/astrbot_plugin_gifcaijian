import re
import io
import asyncio
import aiohttp
from PIL import Image as PILImage, ImageSequence
from astrbot.api.event import filter
from astrbot.api.all import *
import astrbot.api.message_components as Comp

@register(
    "astrbot_plugin_gifcaijian",
    "shskjw",
    "可以裁剪和合并gif",
    "1.0.1",
    "https://github.com/shkjw/astrbot_plugin_gifcaijian",
)
class SpriteToGifPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    def _get_image_url(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "get_images"):
            images = event.get_images()
            if images: return images[0].url

        if hasattr(event.message_obj, "message"):
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Reply) and seg.chain:
                    for item in seg.chain:
                        if isinstance(item, Comp.Image) and item.url: return item.url
                        if isinstance(item, dict) and item.get('type') == 'image':
                            return item.get('data', {}).get('url') or item.get('url')
                if isinstance(seg, dict) and seg.get('type') == 'image':
                    return seg.get('data', {}).get('url') or seg.get('url')
                if isinstance(seg, Comp.Image) and seg.url:
                    return seg.url
        return None

    async def _download_image(self, url: str) -> bytes:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=30) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()

    async def _handle_gif_task(self, event: AstrMessageEvent, algorithm_mode: int):
        msg_text = event.message_str
        
        clean_text = msg_text.replace("合成1gif", "").replace("合成2gif", "").replace("合成gif", "")

        rows = 6
        cols = 6
        duration = 0.1
        margin = 0

        grid_match = re.search(r'(\d+)\s*[*x×]\s*(\d+)', clean_text)
        if grid_match:
            rows = int(grid_match.group(1))
            cols = int(grid_match.group(2))
            clean_text = clean_text.replace(grid_match.group(0), " ")
        
        margin_match = re.search(r'(?:边距|margin|gap)\s*(\d+)', clean_text)
        if margin_match:
            margin = int(margin_match.group(1))
            clean_text = clean_text.replace(margin_match.group(0), " ")

        duration_match = re.search(r'(\d+(?:\.\d+)?)', clean_text)
        if duration_match:
            try:
                val = float(duration_match.group(1))
                if 0 < val <= 60:
                    duration = val
            except ValueError:
                pass

        img_url = self._get_image_url(event)
        if not img_url:
            yield event.plain_result("❌ 未检测到图片")
            return

        try:
            margin_info = f", 边距{margin}px" if margin > 0 else ""
            yield event.plain_result(f"⏳ 正在合成(算法{algorithm_mode})... ({rows}x{cols}, 每帧{duration}秒{margin_info})")
            await asyncio.sleep(0.5)
        except Exception:
            return
        
        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 图片下载失败")
            return

        if algorithm_mode == 1:
            result_msg, gif_bytes = await asyncio.to_thread(self.process_mode_1, img_data, rows, cols, duration, margin)
        else:
            result_msg, gif_bytes = await asyncio.to_thread(self.process_mode_2, img_data, rows, cols, duration, margin)
        
        if gif_bytes:
            yield event.chain_result([Comp.Plain(result_msg), Comp.Image.fromBytes(gif_bytes.getvalue())])
        else:
            yield event.plain_result(f"❌ 失败：\n{result_msg}")

    @filter.command("合成1gif")
    async def make_gif_v1(self, event: AstrMessageEvent):
        async for res in self._handle_gif_task(event, 1):
            yield res

    @filter.command("合成2gif")
    async def make_gif_v2(self, event: AstrMessageEvent):
        async for res in self._handle_gif_task(event, 2):
            yield res

    def process_mode_1(self, img_data: bytes, rows: int, cols: int, duration_sec: float, margin: int):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if getattr(img, "is_animated", False): img.seek(0)
            img = img.convert("RGBA")
            width, height = img.size
            
            disposal_method = 0
            extrema = img.getextrema()
            if extrema[3][0] < 255:
                disposal_method = 2
            
            content_width = width - (cols - 1) * margin
            content_height = height - (rows - 1) * margin
            
            cell_width = content_width // cols
            cell_height = content_height // rows
            
            info_msg = f"算法1 | 原图:{width}x{height} | 切割:{rows}行{cols}列 | 边距:{margin}"
            
            if cell_width < 2 or cell_height < 2: 
                return f"⚠️ 计算后单格太小 ({cell_width}x{cell_height})，请检查边距是否过大", None

            frames = []
            for r in range(rows):
                for c in range(cols):
                    left = c * (cell_width + margin)
                    upper = r * (cell_height + margin)
                    right = left + cell_width
                    lower = upper + cell_height
                    
                    frames.append(img.crop((left, upper, right, lower)))

            output = io.BytesIO()
            duration_ms = int(duration_sec * 1000)
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=disposal_method, dither=PILImage.Dither.NONE, optimize=False)
            output.seek(0)
            return f"✅ 合成成功\n{info_msg}", output
        except Exception as e: return f"逻辑异常: {str(e)}", None

    def process_mode_2(self, img_data: bytes, rows: int, cols: int, duration_sec: float, margin: int):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if getattr(img, "is_animated", False): img.seek(0)
            img = img.convert("RGBA")
            width, height = img.size
            
            datas = img.getdata()
            new_data = []
            has_transparency = False
            for item in datas:
                if item[3] < 128:
                    new_data.append((0, 0, 0, 0))
                    has_transparency = True
                else:
                    new_data.append((item[0], item[1], item[2], 255))
            img.putdata(new_data)

            if has_transparency:
                master_palette = img.convert("RGB").quantize(colors=255, method=1, dither=PILImage.Dither.NONE)
            else:
                master_palette = img.convert("RGB").quantize(colors=256, method=1, dither=PILImage.Dither.NONE)

            content_width = width - (cols - 1) * margin
            content_height = height - (rows - 1) * margin
            cell_width = content_width // cols
            cell_height = content_height // rows

            info_msg = f"算法2 | 原图:{width}x{height} | 切割:{rows}行{cols}列 | 边距:{margin}"
            
            if cell_width < 2 or cell_height < 2: 
                return f"⚠️ 计算后单格太小 ({cell_width}x{cell_height})，请检查边距是否过大", None

            frames = []
            for r in range(rows):
                for c in range(cols):
                    left = c * (cell_width + margin)
                    upper = r * (cell_height + margin)
                    right = left + cell_width
                    lower = upper + cell_height

                    box = (left, upper, right, lower)
                    crop_rgba = img.crop(box)
                    frame_p = crop_rgba.convert("RGB").quantize(palette=master_palette, dither=PILImage.Dither.NONE)
                    
                    if has_transparency:
                        mask = crop_rgba.split()[3].point(lambda a: 255 if a < 128 else 0)
                        try:
                            frame_p.paste(255, mask=mask)
                        except:
                            pass
                    frames.append(frame_p)

            output = io.BytesIO()
            duration_ms = int(duration_sec * 1000)
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2, transparency=255 if has_transparency else None, optimize=False)
            output.seek(0)
            return f"✅ 合成成功\n{info_msg}", output
        except Exception as e: return f"逻辑异常: {str(e)}", None

    @filter.regex(r"(?:gif)?(变快|变慢|加速|减速)\s*[*x×]?\s*(\d+\.?\d*)?")
    async def change_gif_speed(self, event: AstrMessageEvent):
        msg_text = event.message_str
        match = re.search(r"(?:gif)?(变快|变慢|加速|减速)\s*[*x×]?\s*(\d+\.?\d*)?", msg_text)
        if not match: return

        action = match.group(1)
        factor = float(match.group(2)) if match.group(2) else 2.0

        if factor <= 0: factor = 2.0
        if factor > 20: factor = 20.0

        speed_ratio = 1 / factor if action in ["变快", "加速"] else factor
        action_text = "加速" if action in ["变快", "加速"] else "减速"
        
        img_url = self._get_image_url(event)
        if not img_url: return 

        try:
            yield event.plain_result(f"⏳ 正在处理 {action_text} {factor}倍...")
            await asyncio.sleep(0.5)
        except Exception:
            return

        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 图片下载失败")
            return

        result_msg, gif_bytes = await asyncio.to_thread(self.process_speed, img_data, speed_ratio)

        if gif_bytes:
            yield event.chain_result([Comp.Plain(result_msg), Comp.Image.fromBytes(gif_bytes.getvalue())])
        else:
            yield event.plain_result(f"❌ 失败：{result_msg}")

    def process_speed(self, img_data: bytes, ratio: float):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if not getattr(img, "is_animated", False): return "这不是GIF", None

            frames = []
            durations = []
            for frame in ImageSequence.Iterator(img):
                new_frame = frame.copy()
                original_duration = frame.info.get('duration', 100)
                new_duration = int(original_duration * ratio)
                if new_duration < 20: new_duration = 20
                durations.append(new_duration)
                frames.append(new_frame)

            if not frames: return "解析失败", None

            output = io.BytesIO()
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2, dither=PILImage.Dither.NONE, optimize=False)
            output.seek(0)
            return "✅ 变速完成", output
        except Exception as e: return f"处理异常: {str(e)}", None

    @filter.command("裁剪")
    async def crop_and_forward(self, event: AstrMessageEvent):
        msg_text = event.message_str
        clean_text = msg_text.replace("裁剪", "")

        rows = 0
        cols = 0
        margin = 0

        grid_match = re.search(r'(\d+)\s*[*x×]\s*(\d+)', clean_text)
        if grid_match:
            rows = int(grid_match.group(1))
            cols = int(grid_match.group(2))
            clean_text = clean_text.replace(grid_match.group(0), " ")
        else:
            yield event.plain_result("❌ 格式错误，请发送如：裁剪 3*3")
            return

        if rows <= 0 or cols <= 0 or rows > 20 or cols > 20:
             yield event.plain_result("⚠️ 行列数必须 >0 且 <=20")
             return

        margin_match = re.search(r'(?:边距|margin|gap)\s*(\d+)', clean_text)
        if margin_match:
            margin = int(margin_match.group(1))

        img_url = self._get_image_url(event)
        if not img_url:
            yield event.plain_result("❌ 请发送图片带指令，或回复一张图片")
            return
        
        try:
            margin_info = f", 边距{margin}px" if margin > 0 else ""
            yield event.plain_result(f"⏳ 正在裁剪为 {rows}行 x {cols}列{margin_info}，请稍候...")
            await asyncio.sleep(1.0) 
        except Exception:
            return

        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 图片下载失败")
            return

        nodes_data = await asyncio.to_thread(self.process_crop, img_data, rows, cols, margin)

        if not nodes_data:
             yield event.plain_result("❌ 裁剪失败：图片太小或计算错误")
             return

        try:
            nodes_list = []
            for idx, img_bytes in enumerate(nodes_data):
                 node = Comp.Node(
                    name=f"图 {idx+1}",
                    content=[Comp.Image.fromBytes(img_bytes)]
                )
                 nodes_list.append(node)
            
            nodes = Comp.Nodes(nodes=nodes_list)
            yield event.chain_result([nodes])
        except Exception as e:
            yield event.plain_result(f"❌ 发送合并消息失败: {e}")

    def process_crop(self, img_data: bytes, rows: int, cols: int, margin: int):
        try:
            pil_img = PILImage.open(io.BytesIO(img_data))
            pil_img = pil_img.convert("RGBA")
            width, height = pil_img.size
            
            content_width = width - (cols - 1) * margin
            content_height = height - (rows - 1) * margin
            cell_width = content_width // cols
            cell_height = content_height // rows

            if cell_width < 1 or cell_height < 1:
                return None

            img_bytes_list = []
            for r in range(rows):
                for c in range(cols):
                    left = c * (cell_width + margin)
                    upper = r * (cell_height + margin)
                    right = left + cell_width
                    lower = upper + cell_height
                    
                    cropped = pil_img.crop((left, upper, right, lower))
                    
                    img_byte_arr = io.BytesIO()
                    cropped.save(img_byte_arr, format='PNG')
                    img_bytes_list.append(img_byte_arr.getvalue())
            
            return img_bytes_list

        except Exception:
            return None
