import asyncio
import os
from part3 import KoreanLearningPipeline

async def main():
    
    sample_text = (
        "이 아름다운 해바라기 조화는 어디서 줴작한 거"
    )
    
    test_output_directory = "./test_output"
    
    print("="*50)
    print("한국어 단어장 파이프라인 테스트를 시작합니다.")
    print(f"입력 텍스트: {sample_text}")
    print("="*50)
    
    pipeline = KoreanLearningPipeline(
        term_api_key="",
        dict_api_key="", 
        model_name="gemma3:4b" 
    )
    
    try:
        results = await pipeline.run(stt_text=sample_text, output_dir=test_output_directory)
        
        print("\n" + "="*50)
        print("파이프라인 테스트가 정상적으로 완료되었습니다.")
        print(f"총 {len(results)}개의 단어 데이터가 처리되었습니다.")
        print(f"결과물 확인: {os.path.abspath(test_output_directory)}")
        print("="*50)
        
    except Exception as e:
        print(f"\n테스트 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())