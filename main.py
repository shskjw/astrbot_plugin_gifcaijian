import re
import io
import os
import asyncio
import aiohttp
import tempfile
from PIL import Image as PILImage, ImageSequence
from astrbot.api.event import filter
from astrbot.api.all import *
from astrbot.api import logger
import astrbot.api.message_components as Comp

try:
    import imageio
except ImportError:
    imageio = None

@register(
    "astrbot_plugin_gifcaijian",
    "shskjw",
    "1.3.5",
    "可以裁剪和合并gif 分解等功能（后续慢慢润化）",
    "https://github.com/shkjw/astrbot_plugin_gifcaijian",
)
class SpriteToGifPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.cfg = config if config is not None else {}
        
        if imageio is None:
            logger.warning("插件[astrbot_plugin_gifcaijian]检测到缺少 imageio 库。请运行 pip install imageio[ffmpeg]")

    # --- 核心工具：统一保存动画 (支持 GIF/APNG/WebP) ---
    def _save_animation(self, output: io.BytesIO, frames: list, duration_ms: int, loop: int = 0):
        fmt = self.cfg.get('output_format', 'GIF').upper()
        
        # 1. GIF 格式 (兼容性最好)
        if fmt == 'GIF':
            frames[0].save(
                output, format='GIF', save_all=True, append_images=frames[1:], 
                duration=duration_ms, loop=loop, optimize=True, disposal=2
            )
            return

        # 2. APNG 格式 (画质好，支持半透明)
        elif fmt == 'APNG':
            frames[0].save(
                output, format='PNG', save_all=True, append_images=frames[1:], 
                duration=duration_ms, loop=loop, optimize=True, default_image=True
            )
            return

        # 3. WebP 格式 (体积最小，推荐)
        elif fmt == 'WEBP':
            frames[0].save(
                output, format='WEBP', save_all=True, append_images=frames[1:], 
                duration=duration_ms, loop=loop, method=3, quality=80
            )
            return

        # 默认回退到 GIF
        frames[0].save(
            output, format='GIF', save_all=True, append_images=frames[1:], 
            duration=duration_ms, loop=loop, optimize=True, disposal=2
        )

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

    def _get_video_source(self, event: AstrMessageEvent) -> str:
        candidates = []
        def extract_from_item(item):
            # URL
            url = getattr(item, 'url', None)
            if not url and isinstance(item, dict):
                url = item.get('data', {}).get('url') or item.get('url')
            if url and isinstance(url, str) and url.startswith('http'):
                return 100, url
            # Path
            path = getattr(item, 'path', None)
            if not path and isinstance(item, dict):
                path = item.get('data', {}).get('path') or item.get('path')
            if path and isinstance(path, str) and os.path.isabs(path) and os.path.exists(path):
                return 90, path
            # File
            file_info = getattr(item, 'file', None)
            if not file_info and isinstance(item, dict):
                file_info = item.get('data', {}).get('file') or item.get('file')
            if file_info and isinstance(file_info, str):
                return 50, file_info
            return 0, None

        items_to_check = []
        if hasattr(event, "get_videos"):
            videos = event.get_videos()
            if videos: items_to_check.extend(videos)

        if hasattr(event.message_obj, "message"):
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Reply) and seg.chain:
                    items_to_check.extend(seg.chain)
                elif isinstance(seg, (Comp.Video, dict)):
                    items_to_check.append(seg)
                elif isinstance(seg, dict) and seg.get('type') == 'video':
                    items_to_check.append(seg)

        for item in items_to_check:
            score, val = extract_from_item(item)
            if val: candidates.append((score, val))

        if not candidates: return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    async def _resolve_file_via_api(self, event: AstrMessageEvent, file_id: str) -> str:
        try:
            logger.info(f"尝试通过API解析文件ID: {file_id}")
            res = await event.bot.api.call_action("get_file", file_id=file_id)
            if not res or not isinstance(res, dict): return None
            
            url = res.get('url')
            if url and url.startswith('http'): return url
                
            path = res.get('file')
            if path and os.path.exists(path): return path
            return url or path
        except Exception as e:
            logger.warning(f"API解析文件失败: {e}")
            return None

    def _parse_video_args(self, text: str):
        default_scale = self.cfg.get('default_scale', 0.3)
        default_fps = self.cfg.get('default_fps', 10)

        params = {
            'start': 0.0,
            'end': None,
            'fps': default_fps,
            'step': 1,
            'scale': default_scale,
            'force_step': False
        }
        
        # 1. 时间区间 (0.0s-10.0s)
        time_range = re.search(r'(\d+(?:\.\d+)?)[sS]?\s*[-~]\s*(\d+(?:\.\d+)?)[sS]?', text)
        if time_range:
            params['start'] = float(time_range.group(1))
            params['end'] = float(time_range.group(2))
            text = text.replace(time_range.group(0), " ")
        else:
            start_match = re.search(r'(?:开始|start)\s*(\d+(?:\.\d+)?)', text)
            dur_match = re.search(r'(?:时长|len|time)\s*(\d+(?:\.\d+)?)', text)
            if start_match: 
                params['start'] = float(start_match.group(1))
            if dur_match:
                duration = float(dur_match.group(1))
                params['end'] = params['start'] + duration

        # 2. 抽帧: 1/2 或 2/1 (取最大值)
        step_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if step_match:
            n1 = int(step_match.group(1))
            n2 = int(step_match.group(2))
            step_val = max(n1, n2)
            if step_val > 0:
                params['step'] = step_val
                params['fps'] = None
                params['force_step'] = True
            text = text.replace(step_match.group(0), " ")
        else:
            fps_match = re.search(r'(?:fps|帧率)\s*(\d+)', text)
            if fps_match:
                params['fps'] = int(fps_match.group(1))

        # 3. 缩放
        scale_match = re.search(r'\b(0\.\d+|1\.0)\b', text)
        if scale_match:
            params['scale'] = float(scale_match.group(1))

        if params['scale'] < 0.1: params['scale'] = 0.1
        if params['scale'] > 1.0: params['scale'] = 1.0

        return params

    # --- 核心处理逻辑 ---
    def _process_gif_core(self, video_path: str, params: dict, max_colors: int = 256):
        try:
            reader = imageio.get_reader(video_path, format='FFMPEG')
            meta = reader.get_meta_data()
            video_duration = meta.get('duration', 100)
            src_fps = meta.get('fps', 30) or 30
            
            start_t = params['start']
            if params['end'] is None:
                end_t = video_duration
            else:
                end_t = params['end']
            
            max_dur_conf = self.cfg.get('max_gif_duration', 10.0)
            if (end_t - start_t) > max_dur_conf:
                end_t = start_t + max_dur_conf
                warn_msg = f"(限时{max_dur_conf}s)"
            else:
                warn_msg = ""

            end_t = min(end_t, video_duration)
            if start_t >= video_duration:
                return None, f"❌ 开始时间超限", 0

            # 抽帧逻辑
            step = 1
            target_fps = 0
            if params.get('force_step'):
                step = params['step']
                target_fps = src_fps / step
            elif params.get('fps'):
                target_fps = params['fps']
                if target_fps > src_fps: target_fps = src_fps
                step = max(1, int(src_fps / target_fps))
            else:
                step = 3
                target_fps = src_fps / step

            frames = []
            output_fmt = self.cfg.get('output_format', 'GIF').upper()

            for i, frame in enumerate(reader):
                current_time = i / src_fps
                if current_time < start_t: continue
                if current_time > end_t: break
                    
                if i % step == 0:
                    pil_img = PILImage.fromarray(frame)
                    
                    # 缩放
                    w, h = pil_img.size
                    new_w = int(w * params['scale'])
                    new_h = int(h * params['scale'])
                    pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BILINEAR)
                    
                    # 仅在输出格式为 GIF 时进行颜色量化
                    if output_fmt == 'GIF' and max_colors < 256:
                        pil_img = pil_img.quantize(colors=max_colors, method=1, dither=PILImage.Dither.FLOYDSTEINBERG)
                    
                    frames.append(pil_img)
                    
                if len(frames) > 400:
                    warn_msg += " [帧数截断]"
                    break
            
            reader.close()
            
            if not frames:
                return None, "❌ 无有效帧", 0
                
            output = io.BytesIO()
            duration_ms = int(1000 / target_fps) if target_fps > 0 else 100
            
            # 使用统一保存方法
            self._save_animation(output, frames, duration_ms, loop=0)
            output.seek(0)
            
            size_mb = output.getbuffer().nbytes / 1024 / 1024
            
            info = f"时间:{start_t}-{end_t:.1f}s {warn_msg}\n"
            info += f"格式:{output_fmt} | FPS:{target_fps:.1f}\n"
            info += f"缩放:{params['scale']} | 体积:{size_mb:.2f}MB"
            
            return output, info, size_mb

        except Exception as e:
            return None, f"内部错误: {repr(e)}", 0

    # --- 工作线程 wrapper ---
    def _worker_video_to_gif_wrapper(self, video_path: str, params: dict):
        if imageio is None:
            return "❌ 缺少依赖库 imageio", None
            
        max_colors = self.cfg.get('gif_max_colors', 256)
        
        # 第一次尝试
        gif_io, msg, size_mb = self._process_gif_core(video_path, params, max_colors)
        if not gif_io: return msg, None
            
        # 智能重试 (仅针对 GIF，APNG/WebP 暂不自动降级因为通常是为了画质)
        # 如果体积 > 10MB 且是 GIF，尝试压缩
        output_fmt = self.cfg.get('output_format', 'GIF').upper()
        if size_mb > 10.0 and output_fmt == 'GIF':
            new_params = params.copy()
            new_msg_prefix = f"⚠️ 初次体积{size_mb:.1f}MB过大，自动压缩中...\n"
            
            new_colors = 128 if max_colors > 128 else 64
            new_params['scale'] = round(params['scale'] * 0.8, 2)
            if new_params['scale'] < 0.1: new_params['scale'] = 0.1
            
            retry_io, retry_msg, retry_size = self._process_gif_core(video_path, new_params, new_colors)
            if retry_io and retry_size < size_mb:
                return new_msg_prefix + retry_msg, retry_io
            else:
                return f"⚠️ 压缩失败({retry_size:.1f}MB)，原版:\n" + msg, gif_io
        
        return "✅ 转换成功\n" + msg, gif_io

    @filter.command("视频转gif")
    async def video_to_gif_cmd(self, event: AstrMessageEvent):
        if imageio is None:
            yield event.plain_result("❌ 无法使用此功能：服务器缺少 imageio 库。")
            return

        msg_text = event.message_str.replace("视频转gif", "")
        params = self._parse_video_args(msg_text)
        
        raw_source = self._get_video_source(event)
        if not raw_source:
            yield event.plain_result("❌ 请回复一个视频或发送视频链接。")
            return

        valid_source = None
        if raw_source.startswith("http") or os.path.exists(raw_source):
            valid_source = raw_source
        else:
            yield event.plain_result("⏳ 正在请求视频地址...")
            valid_source = await self._resolve_file_via_api(event, raw_source)
            if not valid_source:
                yield event.plain_result(f"❌ 无法解析视频地址: {raw_source}")
                return

        fmt = self.cfg.get('output_format', 'GIF')
        time_info = f"{params['start']}s-" + (f"{params['end']}s" if params['end'] else "末尾")
        yield event.plain_result(f"⏳ 任务已接收 ({fmt})\n区间: {time_info}\n缩放: {params['scale']}")
        
        tmp_path = ""
        is_temp_file = False

        try:
            if valid_source.startswith("http"):
                max_size = self.cfg.get('max_video_size_mb', 50.0) * 1024 * 1024
                
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    is_temp_file = True
                
                headers = {"User-Agent": "Mozilla/5.0"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(valid_source, headers=headers, timeout=120) as resp:
                        if resp.status != 200:
                            yield event.plain_result(f"❌ 下载失败 HTTP {resp.status}")
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                            return
                        
                        content_len = resp.headers.get('Content-Length')
                        if content_len and int(content_len) > max_size:
                            yield event.plain_result(f"❌ 视频超过大小限制")
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                            return

                        with open(tmp_path, 'wb') as f:
                            f.write(await resp.read())
            else:
                tmp_path = valid_source
                is_temp_file = False

            result_msg, gif_bytes = await asyncio.to_thread(self._worker_video_to_gif_wrapper, tmp_path, params)
            
            if is_temp_file and os.path.exists(tmp_path): 
                os.remove(tmp_path)
                
            if gif_bytes:
                yield event.chain_result([Comp.Plain(result_msg), Comp.Image.fromBytes(gif_bytes.getvalue())])
            else:
                yield event.plain_result(result_msg)
                
        except Exception as e:
            if is_temp_file and tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
            yield event.plain_result(f"❌ 处理异常: {repr(e)}")

    # --- 其他原有功能 ---
    def _parse_margins(self, text: str):
        margins = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        pattern = r'边距\s*([上下左右])?边?\s*(\d+)'
        matches = re.findall(pattern, text)
        for direction, amount_str in matches:
            try:
                amount = int(amount_str)
                if not direction:
                    for k in margins: margins[k] += amount
                elif direction == '上': margins['top'] += amount
                elif direction == '下': margins['bottom'] += amount
                elif direction == '左': margins['left'] += amount
                elif direction == '右': margins['right'] += amount
            except ValueError: pass
        clean_text = re.sub(pattern, " ", text)
        return clean_text, margins

    def _crop_image_data(self, img_data: bytes, margins: dict) -> tuple[bytes, str]:
        if all(v == 0 for v in margins.values()): return img_data, ""
        try:
            img = PILImage.open(io.BytesIO(img_data)).convert("RGBA")
            w, h = img.size
            l, u, r, d = margins['left'], margins['top'], w - margins['right'], h - margins['bottom']
            if l >= r or u >= d: return img_data, f"\n⚠️ 边距无效: {w}x{h} -> {l},{u},{r},{d}"
            output = io.BytesIO()
            img.crop((l, u, r, d)).save(output, format='PNG')
            return output.getvalue(), f"\n✂️ 已裁边距: 上{margins['top']} 下{margins['bottom']} 左{margins['left']} 右{margins['right']}"
        except Exception as e: return img_data, f"\n⚠️ 边距裁剪出错: {e}"

    async def _download_image(self, url: str) -> bytes:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=30) as resp:
                    if resp.status != 200: return None
                    return await resp.read()
            except: return None

    async def _handle_gif_task(self, event: AstrMessageEvent, algorithm_mode: int):
        msg_text = event.message_str
        clean_text, margins = self._parse_margins(msg_text)
        clean_text = clean_text.replace("合成1gif", "").replace("合成2gif", "").replace("合成gif", "")
        rows, cols, duration = 6, 6, 0.1
        grid_match = re.search(r'(\d+)\s*[*x×]\s*(\d+)', clean_text)
        if grid_match:
            rows, cols = int(grid_match.group(1)), int(grid_match.group(2))
            clean_text = clean_text.replace(grid_match.group(0), " ")
        dur_match = re.search(r'(\d+(?:\.\d+)?)', clean_text)
        if dur_match:
            try:
                val = float(dur_match.group(1))
                if 0 < val <= 60: duration = val
            except: pass
        img_url = self._get_image_url(event)
        if not img_url:
            yield event.plain_result("❌ 未检测到图片")
            return
        yield event.plain_result(f"⏳ 正在合成(算法{algorithm_mode})... ({rows}x{cols}, 每帧{duration}s)")
        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 图片下载失败")
            return
        img_data, crop_msg = await asyncio.to_thread(self._crop_image_data, img_data, margins)
        func = self.process_mode_1 if algorithm_mode == 1 else self.process_mode_2
        res_msg, gif_bytes = await asyncio.to_thread(func, img_data, rows, cols, duration)
        if gif_bytes:
            yield event.chain_result([Comp.Plain(res_msg + crop_msg), Comp.Image.fromBytes(gif_bytes.getvalue())])
        else:
            yield event.plain_result(f"❌ 失败：\n{res_msg}")

    @filter.command("合成1gif")
    async def make_gif_v1(self, event: AstrMessageEvent):
        async for res in self._handle_gif_task(event, 1): yield res

    @filter.command("合成2gif")
    async def make_gif_v2(self, event: AstrMessageEvent):
        async for res in self._handle_gif_task(event, 2): yield res

    def process_mode_1(self, img_data: bytes, rows: int, cols: int, duration_sec: float):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if getattr(img, "is_animated", False): img.seek(0)
            img = img.convert("RGBA")
            w, h = img.size
            cw, ch = w // cols, h // rows
            if cw < 2 or ch < 2: return f"⚠️ 单格太小 ({cw}x{ch})", None
            frames = []
            for r in range(rows):
                for c in range(cols):
                    frames.append(img.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch)))
            output = io.BytesIO()
            self._save_animation(output, frames, int(duration_sec*1000), loop=0)
            output.seek(0)
            return f"✅ 合成成功\n算法1 | {w}x{h} | {rows}行{cols}列", output
        except Exception as e: return f"逻辑异常: {e}", None

    def process_mode_2(self, img_data: bytes, rows: int, cols: int, duration_sec: float):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if getattr(img, "is_animated", False): img.seek(0)
            img = img.convert("RGBA")
            datas = img.getdata()
            new_data = [(0,0,0,0) if item[3] < 128 else (item[0],item[1],item[2],255) for item in datas]
            img.putdata(new_data)
            has_trans = any(d[3] == 0 for d in new_data)
            master_pal = img.convert("RGB").quantize(colors=255 if has_trans else 256, method=1)
            w, h = img.size
            cw, ch = w // cols, h // rows
            if cw < 2 or ch < 2: return f"⚠️ 单格太小 ({cw}x{ch})", None
            frames = []
            for r in range(rows):
                for c in range(cols):
                    crop = img.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch))
                    frame = crop.convert("RGB").quantize(palette=master_pal)
                    if has_trans:
                        mask = crop.split()[3].point(lambda a: 255 if a < 128 else 0)
                        frame.paste(255, mask=mask)
                    frames.append(frame)
            output = io.BytesIO()
            fmt = self.cfg.get('output_format', 'GIF').upper()
            if fmt == 'GIF':
                 frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], 
                           duration=int(duration_sec*1000), loop=0, disposal=2, 
                           transparency=255 if has_trans else None, optimize=True)
            else:
                 self._save_animation(output, frames, int(duration_sec*1000), loop=0)
            output.seek(0)
            return f"✅ 合成成功\n算法2 | {w}x{h} | {rows}行{cols}列", output
        except Exception as e: return f"逻辑异常: {e}", None

    @filter.regex(r"(?:gif)?(变快|变慢|加速|减速)\s*[*x×]?\s*(\d+\.?\d*)?")
    async def change_gif_speed(self, event: AstrMessageEvent):
        msg = event.message_str
        match = re.search(r"(?:gif)?(变快|变慢|加速|减速)\s*[*x×]?\s*(\d+\.?\d*)?", msg)
        if not match: return
        action, factor = match.group(1), float(match.group(2) or 2.0)
        factor = max(0.1, min(factor, 20.0))
        ratio = 1/factor if action in ["变快", "加速"] else factor
        img_url = self._get_image_url(event)
        if not img_url: return
        yield event.plain_result(f"⏳ 正在处理 {action} {factor}倍...")
        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 下载失败")
            return
        res_msg, gif_bytes = await asyncio.to_thread(self.process_speed, img_data, ratio)
        if gif_bytes:
            yield event.chain_result([Comp.Plain(res_msg), Comp.Image.fromBytes(gif_bytes.getvalue())])
        else:
            yield event.plain_result(f"❌ 失败：{res_msg}")

    def process_speed(self, img_data: bytes, ratio: float):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if not getattr(img, "is_animated", False): return "这不是GIF", None
            frames, durs = [], []
            for frame in ImageSequence.Iterator(img):
                durs.append(max(20, int(frame.info.get('duration', 100) * ratio)))
                frames.append(frame.copy())
            output = io.BytesIO()
            # 变速功能默认保存为GIF以确保兼容性
            frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:], 
                           duration=durs, loop=0, disposal=2, optimize=True)
            output.seek(0)
            return "✅ 变速完成", output
        except Exception as e: return f"异常: {e}", None

    def _worker_crop_grid(self, img_data: bytes, margins: dict, rows: int, cols: int):
        img_data, crop_msg = self._crop_image_data(img_data, margins)
        try:
            img = PILImage.open(io.BytesIO(img_data)).convert("RGBA")
            w, h = img.size
            cw, ch = w // cols, h // rows
            if cw < 1 or ch < 1: return f"❌ 图片太小 {crop_msg}", None
            res_list = []
            for r in range(rows):
                for c in range(cols):
                    out = io.BytesIO()
                    img.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch)).save(out, format='PNG')
                    res_list.append(out.getvalue())
            return crop_msg, res_list
        except Exception as e: return f"❌ 出错: {e}", None

    @filter.command("裁剪")
    async def crop_and_forward(self, event: AstrMessageEvent):
        clean, margins = self._parse_margins(event.message_str)
        match = re.search(r'(\d+)\s*[*x×]\s*(\d+)', clean)
        rows, cols = (int(match.group(1)), int(match.group(2))) if match else (1, 1)
        if rows > 20 or cols > 20:
            yield event.plain_result("⚠️ 行列数过大")
            return
        img_url = self._get_image_url(event)
        if not img_url:
            yield event.plain_result("❌ 请发送图片")
            return
        yield event.plain_result("⏳ 处理中...")
        img_data = await self._download_image(img_url)
        if not img_data:
            yield event.plain_result("❌ 下载失败")
            return
        msg, bytes_list = await asyncio.to_thread(self._worker_crop_grid, img_data, margins, rows, cols)
        if not bytes_list:
            yield event.plain_result(msg)
            return
        nodes = [Comp.Node(name="裁剪", content=[Comp.Plain(f"结果 {rows}x{cols}{msg}")])]
        for b in bytes_list:
            nodes.append(Comp.Node(name="裁剪", content=[Comp.Image.fromBytes(b)]))
        yield event.chain_result([Comp.Nodes(nodes=nodes)])

    @filter.command("gif分解")
    async def decompose_gif(self, event: AstrMessageEvent):
        img_url = self._get_image_url(event)
        if not img_url:
            yield event.plain_result("❌ 请发送GIF")
            return
        yield event.plain_result("⏳ 分解中...")
        img_data = await self._download_image(img_url)
        frames = await asyncio.to_thread(self._worker_decompose, img_data)
        if isinstance(frames, str):
            yield event.plain_result(frames)
            return
        nodes = [Comp.Node(name="GIF助手", content=[Comp.Plain(f"第{i+1}帧"), Comp.Image.fromBytes(b)]) for i, b in enumerate(frames)]
        yield event.chain_result([Comp.Nodes(nodes=nodes)])

    def _worker_decompose(self, img_data: bytes):
        try:
            img = PILImage.open(io.BytesIO(img_data))
            if not getattr(img, "is_animated", False): return "⚠️ 不是GIF动画"
            frames = []
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if i >= 100: break
                out = io.BytesIO()
                frame.copy().convert("RGBA").save(out, format='PNG')
                frames.append(out.getvalue())
            return frames
        except Exception as e: return f"❌ 出错: {e}"
