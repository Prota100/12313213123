import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
import yt_dlp
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
import re
from konlpy.tag import Okt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# .env 파일 로드
load_dotenv()

# API 키 설정
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def get_video_info(video_url):
    """유튜브 영상 정보 가져오기"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            'title': info.get('title', ''),
            'description': info.get('description', ''),
            'duration': info.get('duration', 0)
        }

def get_youtube_transcript(video_url):
    """유튜브 자막 가져오기"""
    try:
        # video_id 추출
        if 'youtu.be' in video_url:
            video_id = video_url.split('/')[-1].split('?')[0]
        else:
            video_id = video_url.split('v=')[1].split('&')[0]
            
        # 자막 가져오기
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        
        # 자막 텍스트 처리
        processed_lines = []
        current_line = ""
        
        for entry in transcript:
            text = entry['text'].strip()
            # 문장이 끝나는 구두점으로 끝나는 경우
            if text.endswith(('.', '!', '?', '...', '"', "'", '」', '다.', '요.', '죠.', '까.', '니다.')):
                current_line += text + "\n"
                if current_line.strip():
                    processed_lines.append(current_line.strip())
                current_line = ""
            else:
                current_line += text + " "
        
        # 마지막 라인 처리
        if current_line.strip():
            processed_lines.append(current_line.strip())
        
        return "\n".join(processed_lines)
    except Exception as e:
        print(f"자막을 가져오는 중 오류 발생: {e}")
        return None

def extract_keywords(text):
    """키워드 추출"""
    try:
        okt = Okt()
        # 명사 추출
        nouns = okt.nouns(text)
        # 빈도수 계산
        from collections import Counter
        count = Counter(nouns)
        # 상위 10개 키워드 추출
        keywords = [word for word, count in count.most_common(10)]
        return ", ".join(keywords)
    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {e}")
        return None

def summarize_with_sumy(text, sentences_count=5):
    """Sumy를 사용한 요약"""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("korean"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return "\n".join([str(sentence) for sentence in summary])
    except Exception as e:
        print(f"Sumy 요약 중 오류 발생: {e}")
        return None

def summarize_text(text, video_info):
    """텍스트 요약"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Sumy로 기본 요약 생성
    sumy_summary = summarize_with_sumy(text)
    
    # 키워드 추출
    keywords = extract_keywords(text)
    
    prompt = f"""
다음 유튜브 영상의 내용을 체계적이고 자세하게 분석해주세요:

제목: {video_info['title']}
영상 길이: {video_info['duration']}초

기본 요약:
{sumy_summary}

키워드:
{keywords}

전체 자막 내용:
{text}

다음 형식으로 상세하게 작성해주세요:

1. 영상 개요
   - 영상의 주요 주제와 목적
   - 대상 시청자층
   - 영상의 전체적인 흐름

2. 핵심 내용 분석
   - 주요 주제별 상세 설명 (3-4개 섹션)
   - 각 주제의 중요 포인트
   - 구체적인 예시나 사례

3. 논리적 구조
   - 영상의 전개 방식
   - 주장과 근거의 연결 관계
   - 핵심 메시지의 전달 방식

4. 심층 분석
   - 영상의 강점과 특징
   - 기술적/학술적 내용의 상세 설명
   - 시사점과 의의

5. 종합 결론
   - 영상의 핵심 메시지
   - 시청자에게 전달하고자 하는 최종 메시지
   - 추가 학습이나 고려사항

각 섹션은 명확한 소제목과 함께 작성하고, 내용은 구체적이고 이해하기 쉽게 서술해주세요.
"""
    
    try:
        print("\n상세 분석 생성 중...", flush=True)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            print(chunk.text, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"분석 생성 중 오류 발생: {e}")

def get_video_id(url):
    """YouTube URL에서 video_id 추출"""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url:
        return url.split('v=')[1].split('&')[0]
    return None

def get_video_comments(video_url):
    """YouTube Data API를 사용하여 댓글 가져오기"""
    comments = []
    try:
        # YouTube API 클라이언트 생성
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # video_id 추출
        video_id = get_video_id(video_url)
        if not video_id:
            print("유효한 YouTube URL이 아닙니다.")
            return comments
        
        # 댓글 스레드 가져오기
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # 최대 100개 댓글
            order="relevance"  # 관련성 순으로 정렬
        )
        
        while request:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'publishedAt': comment['publishedAt']
                })
            
            # 다음 페이지가 있는지 확인
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response['nextPageToken'],
                    order="relevance"
                )
            else:
                break
            
            # API 할당량을 고려하여 잠시 대기
            time.sleep(0.5)
            
    except HttpError as e:
        print(f"API 오류 발생: {e}")
    except Exception as e:
        print(f"댓글 가져오기 중 오류 발생: {e}")
    
    return comments

def summarize_comments(comments):
    """댓글 요약"""
    if not comments:
        return "댓글을 가져올 수 없습니다."
    
    # 댓글을 좋아요 수로 정렬
    sorted_comments = sorted(comments, key=lambda x: x['likes'], reverse=True)
    
    # 상위 3개 댓글만 선택
    top_comments = sorted_comments[:3]
    
    # 실제 댓글 출력
    print("\n실제 댓글 (상위 3개):")
    for i, comment in enumerate(top_comments, 1):
        print(f"{i}. {comment['text']} (좋아요: {comment['likes']})")
    
    # 댓글 텍스트 준비
    comments_text = "\n".join([
        f"- {comment['text']} (좋아요: {comment['likes']})"
        for comment in top_comments
    ])
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
다음은 유튜브 영상의 상위 3개 댓글입니다:

{comments_text}

이 댓글들의 전반적인 반응을 다음 형식으로 분석해주세요:

1. 주요 관심사
   - 시청자들이 가장 많이 언급한 주제
   - 공통적인 의견이나 반응

2. 감정 분석
   - 긍정적인 반응
   - 부정적인 반응
   - 중립적인 반응

3. 핵심 메시지
   - 댓글들을 통해 드러나는 주요 메시지
   - 시청자들이 강조하는 포인트

각 섹션은 2-3줄로 간단명료하게 작성해주세요.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"댓글 요약 중 오류 발생: {e}")
        return "댓글 요약을 생성할 수 없습니다."

def main():
    # 입력 확인
    if len(os.sys.argv) < 2:
        print("사용법: python ultimate_simple.py <YouTube_URL>")
        return
    
    video_url = os.sys.argv[1]
    print(f"영상 정보를 가져오는 중: {video_url}")
    
    # 영상 정보 가져오기
    video_info = get_video_info(video_url)
    print(f"제목: {video_info['title']}")
    
    # 자막 가져오기
    transcript = get_youtube_transcript(video_url)
    if not transcript:
        print("자막을 가져올 수 없습니다.")
        return
    
    # 요약 생성
    summarize_text(transcript, video_info)
    
    # 댓글 가져오기 및 요약
    print("\n댓글 분석 중...")
    comments = get_video_comments(video_url)
    comments_summary = summarize_comments(comments)
    print("\n시청자 반응 요약:")
    print(comments_summary)

if __name__ == "__main__":
    main() 