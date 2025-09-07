# Known Issues & Compatibility Problems

## GPT-OSS-120B Transformers Compatibility Issue

**Date:** 2025-09-07  
**Model:** `openai/gpt-oss-120b`  
**Status:** ❌ Failed - Transformers library incompatibility

### Problem
```
ValueError: The checkpoint you are trying to load has model type `gpt_oss` but 
Transformers does not recognize this architecture.
```

### Root Cause
The `gpt_oss` model type is too new for current transformers library version. Model was released recently and transformers hasn't added support yet.

### Failed Instance Details
- **Instance ID:** 8xi2k3ubhbaxrt
- **SSH:** root@216.81.245.26:37394
- **GPU:** 1x H100 80GB
- **Deployment:** Successful (fast code push worked)
- **Analysis:** Failed at model loading step

### Potential Solutions
1. **Update transformers:**
   ```bash
   pip install --upgrade transformers
   # or bleeding edge:
   pip install git+https://github.com/huggingface/transformers.git
   ```

2. **Wait for official support** - transformers team may add `gpt_oss` support soon

3. **Custom model loader** - implement manual loading if needed

### Workaround
Use **DeepSeek-V3.1** instead:
- Similar ultra-sparse ratio (5.5% vs 4.4%)
- Proven transformers compatibility 
- More recent (Aug 2025)
- 37B active params vs 5.1B (good research comparison)

### Next Steps
- ☐ Try updated transformers version
- ☐ Monitor transformers releases for gpt_oss support
- ☐ Consider custom model loading implementation
- ✅ Continue with GLM-4.5-Air as next target

---

## Other Compatibility Notes

### Successful Models
- ✅ **Qwen/Qwen3-30B-A3B** - Full compatibility, 24 outliers found
- 🔄 **Analysis pipeline** - Memory-optimized extraction working perfectly

### Deployment Optimizations Working
- ✅ Fast code push (seconds vs 15+ minutes)
- ✅ Multi-GPU provisioning 
- ✅ Memory-optimized processing
- ✅ Bandwidth-optimized sync