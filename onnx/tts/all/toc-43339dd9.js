// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="index.html"><strong aria-hidden="true">1.</strong> TTS models</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese-English/index.html"><strong aria-hidden="true">2.</strong> Chinese+English</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese-English/matcha-icefall-zh-en.html"><strong aria-hidden="true">2.1.</strong> matcha-icefall-zh-en</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese-English/kokoro-multi-lang-v1_0.html"><strong aria-hidden="true">2.2.</strong> kokoro-multi-lang-v1_0</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese-English/kokoro-multi-lang-v1_1.html"><strong aria-hidden="true">2.3.</strong> kokoro-multi-lang-v1_1</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/index.html"><strong aria-hidden="true">3.</strong> Arabic</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/vits-piper-ar_JO-SA_dii-high.html"><strong aria-hidden="true">3.1.</strong> vits-piper-ar_JO-SA_dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/vits-piper-ar_JO-SA_miro-high.html"><strong aria-hidden="true">3.2.</strong> vits-piper-ar_JO-SA_miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/vits-piper-ar_JO-SA_miro_V2-high.html"><strong aria-hidden="true">3.3.</strong> vits-piper-ar_JO-SA_miro_V2-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/vits-piper-ar_JO-kareem-low.html"><strong aria-hidden="true">3.4.</strong> vits-piper-ar_JO-kareem-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Arabic/vits-piper-ar_JO-kareem-medium.html"><strong aria-hidden="true">3.5.</strong> vits-piper-ar_JO-kareem-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Catalan/index.html"><strong aria-hidden="true">4.</strong> Catalan</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Catalan/vits-piper-ca_ES-upc_ona-medium.html"><strong aria-hidden="true">4.1.</strong> vits-piper-ca_ES-upc_ona-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Catalan/vits-piper-ca_ES-upc_ona-x_low.html"><strong aria-hidden="true">4.2.</strong> vits-piper-ca_ES-upc_ona-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Catalan/vits-piper-ca_ES-upc_pau-x_low.html"><strong aria-hidden="true">4.3.</strong> vits-piper-ca_ES-upc_pau-x_low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese/index.html"><strong aria-hidden="true">5.</strong> Chinese</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Chinese/matcha-icefall-zh-baker.html"><strong aria-hidden="true">5.1.</strong> matcha-icefall-zh-baker</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Czech/index.html"><strong aria-hidden="true">6.</strong> Czech</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Czech/vits-piper-cs_CZ-jirka-low.html"><strong aria-hidden="true">6.1.</strong> vits-piper-cs_CZ-jirka-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Czech/vits-piper-cs_CZ-jirka-medium.html"><strong aria-hidden="true">6.2.</strong> vits-piper-cs_CZ-jirka-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Danish/index.html"><strong aria-hidden="true">7.</strong> Danish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Danish/vits-piper-da_DK-talesyntese-medium.html"><strong aria-hidden="true">7.1.</strong> vits-piper-da_DK-talesyntese-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/index.html"><strong aria-hidden="true">8.</strong> Dutch</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_BE-nathalie-medium.html"><strong aria-hidden="true">8.1.</strong> vits-piper-nl_BE-nathalie-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_BE-nathalie-x_low.html"><strong aria-hidden="true">8.2.</strong> vits-piper-nl_BE-nathalie-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_NL-dii-high.html"><strong aria-hidden="true">8.3.</strong> vits-piper-nl_NL-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_NL-miro-high.html"><strong aria-hidden="true">8.4.</strong> vits-piper-nl_NL-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_NL-pim-medium.html"><strong aria-hidden="true">8.5.</strong> vits-piper-nl_NL-pim-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Dutch/vits-piper-nl_NL-ronnie-medium.html"><strong aria-hidden="true">8.6.</strong> vits-piper-nl_NL-ronnie-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/index.html"><strong aria-hidden="true">9.</strong> English</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/matcha-icefall-en_US-ljspeech.html"><strong aria-hidden="true">9.1.</strong> matcha-icefall-en_US-ljspeech</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/kitten-nano-en-v0_1.html"><strong aria-hidden="true">9.2.</strong> kitten-nano-en-v0_1-fp16</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/kitten-nano-en-v0_2.html"><strong aria-hidden="true">9.3.</strong> kitten-nano-en-v0_2-fp16</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/kitten-mini-en-v0_1.html"><strong aria-hidden="true">9.4.</strong> kitten-mini-en-v0_1-fp16</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/kokoro-en-v0_19.html"><strong aria-hidden="true">9.5.</strong> kokoro-en-v0_19</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-alan-low.html"><strong aria-hidden="true">9.6.</strong> vits-piper-en_GB-alan-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-alan-medium.html"><strong aria-hidden="true">9.7.</strong> vits-piper-en_GB-alan-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-alba-medium.html"><strong aria-hidden="true">9.8.</strong> vits-piper-en_GB-alba-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-aru-medium.html"><strong aria-hidden="true">9.9.</strong> vits-piper-en_GB-aru-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-cori-high.html"><strong aria-hidden="true">9.10.</strong> vits-piper-en_GB-cori-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-cori-medium.html"><strong aria-hidden="true">9.11.</strong> vits-piper-en_GB-cori-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-dii-high.html"><strong aria-hidden="true">9.12.</strong> vits-piper-en_GB-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-jenny_dioco-medium.html"><strong aria-hidden="true">9.13.</strong> vits-piper-en_GB-jenny_dioco-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-miro-high.html"><strong aria-hidden="true">9.14.</strong> vits-piper-en_GB-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-northern_english_male-medium.html"><strong aria-hidden="true">9.15.</strong> vits-piper-en_GB-northern_english_male-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-semaine-medium.html"><strong aria-hidden="true">9.16.</strong> vits-piper-en_GB-semaine-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-southern_english_female-low.html"><strong aria-hidden="true">9.17.</strong> vits-piper-en_GB-southern_english_female-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-southern_english_female-medium.html"><strong aria-hidden="true">9.18.</strong> vits-piper-en_GB-southern_english_female-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-southern_english_male-medium.html"><strong aria-hidden="true">9.19.</strong> vits-piper-en_GB-southern_english_male-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_GB-vctk-medium.html"><strong aria-hidden="true">9.20.</strong> vits-piper-en_GB-vctk-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-amy-low.html"><strong aria-hidden="true">9.21.</strong> vits-piper-en_US-amy-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-amy-medium.html"><strong aria-hidden="true">9.22.</strong> vits-piper-en_US-amy-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-arctic-medium.html"><strong aria-hidden="true">9.23.</strong> vits-piper-en_US-arctic-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-bryce-medium.html"><strong aria-hidden="true">9.24.</strong> vits-piper-en_US-bryce-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-danny-low.html"><strong aria-hidden="true">9.25.</strong> vits-piper-en_US-danny-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-glados-high.html"><strong aria-hidden="true">9.26.</strong> vits-piper-en_US-glados-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-hfc_female-medium.html"><strong aria-hidden="true">9.27.</strong> vits-piper-en_US-hfc_female-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-hfc_male-medium.html"><strong aria-hidden="true">9.28.</strong> vits-piper-en_US-hfc_male-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-joe-medium.html"><strong aria-hidden="true">9.29.</strong> vits-piper-en_US-joe-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-john-medium.html"><strong aria-hidden="true">9.30.</strong> vits-piper-en_US-john-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-kathleen-low.html"><strong aria-hidden="true">9.31.</strong> vits-piper-en_US-kathleen-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-kristin-medium.html"><strong aria-hidden="true">9.32.</strong> vits-piper-en_US-kristin-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-kusal-medium.html"><strong aria-hidden="true">9.33.</strong> vits-piper-en_US-kusal-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-l2arctic-medium.html"><strong aria-hidden="true">9.34.</strong> vits-piper-en_US-l2arctic-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-lessac-high.html"><strong aria-hidden="true">9.35.</strong> vits-piper-en_US-lessac-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-lessac-low.html"><strong aria-hidden="true">9.36.</strong> vits-piper-en_US-lessac-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-lessac-medium.html"><strong aria-hidden="true">9.37.</strong> vits-piper-en_US-lessac-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-libritts-high.html"><strong aria-hidden="true">9.38.</strong> vits-piper-en_US-libritts-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-libritts_r-medium.html"><strong aria-hidden="true">9.39.</strong> vits-piper-en_US-libritts_r-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-ljspeech-high.html"><strong aria-hidden="true">9.40.</strong> vits-piper-en_US-ljspeech-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-ljspeech-medium.html"><strong aria-hidden="true">9.41.</strong> vits-piper-en_US-ljspeech-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-miro-high.html"><strong aria-hidden="true">9.42.</strong> vits-piper-en_US-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-norman-medium.html"><strong aria-hidden="true">9.43.</strong> vits-piper-en_US-norman-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-reza_ibrahim-medium.html"><strong aria-hidden="true">9.44.</strong> vits-piper-en_US-reza_ibrahim-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-ryan-high.html"><strong aria-hidden="true">9.45.</strong> vits-piper-en_US-ryan-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-ryan-low.html"><strong aria-hidden="true">9.46.</strong> vits-piper-en_US-ryan-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-ryan-medium.html"><strong aria-hidden="true">9.47.</strong> vits-piper-en_US-ryan-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="English/vits-piper-en_US-sam-medium.html"><strong aria-hidden="true">9.48.</strong> vits-piper-en_US-sam-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Finnish/index.html"><strong aria-hidden="true">10.</strong> Finnish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Finnish/vits-piper-fi_FI-harri-low.html"><strong aria-hidden="true">10.1.</strong> vits-piper-fi_FI-harri-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Finnish/vits-piper-fi_FI-harri-medium.html"><strong aria-hidden="true">10.2.</strong> vits-piper-fi_FI-harri-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/index.html"><strong aria-hidden="true">11.</strong> French</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-gilles-low.html"><strong aria-hidden="true">11.1.</strong> vits-piper-fr_FR-gilles-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-miro-high.html"><strong aria-hidden="true">11.2.</strong> vits-piper-fr_FR-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-siwis-low.html"><strong aria-hidden="true">11.3.</strong> vits-piper-fr_FR-siwis-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-siwis-medium.html"><strong aria-hidden="true">11.4.</strong> vits-piper-fr_FR-siwis-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-tjiho-model1.html"><strong aria-hidden="true">11.5.</strong> vits-piper-fr_FR-tjiho-model1</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-tjiho-model2.html"><strong aria-hidden="true">11.6.</strong> vits-piper-fr_FR-tjiho-model2</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-tjiho-model3.html"><strong aria-hidden="true">11.7.</strong> vits-piper-fr_FR-tjiho-model3</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-tom-medium.html"><strong aria-hidden="true">11.8.</strong> vits-piper-fr_FR-tom-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="French/vits-piper-fr_FR-upmc-medium.html"><strong aria-hidden="true">11.9.</strong> vits-piper-fr_FR-upmc-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Georgian/index.html"><strong aria-hidden="true">12.</strong> Georgian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Georgian/vits-piper-ka_GE-natia-medium.html"><strong aria-hidden="true">12.1.</strong> vits-piper-ka_GE-natia-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/index.html"><strong aria-hidden="true">13.</strong> German</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-dii-high.html"><strong aria-hidden="true">13.1.</strong> vits-piper-de_DE-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-eva_k-x_low.html"><strong aria-hidden="true">13.2.</strong> vits-piper-de_DE-eva_k-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados-high.html"><strong aria-hidden="true">13.3.</strong> vits-piper-de_DE-glados-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados-low.html"><strong aria-hidden="true">13.4.</strong> vits-piper-de_DE-glados-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados-medium.html"><strong aria-hidden="true">13.5.</strong> vits-piper-de_DE-glados-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados_turret-high.html"><strong aria-hidden="true">13.6.</strong> vits-piper-de_DE-glados_turret-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados_turret-low.html"><strong aria-hidden="true">13.7.</strong> vits-piper-de_DE-glados_turret-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-glados_turret-medium.html"><strong aria-hidden="true">13.8.</strong> vits-piper-de_DE-glados_turret-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-karlsson-low.html"><strong aria-hidden="true">13.9.</strong> vits-piper-de_DE-karlsson-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-kerstin-low.html"><strong aria-hidden="true">13.10.</strong> vits-piper-de_DE-kerstin-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-miro-high.html"><strong aria-hidden="true">13.11.</strong> vits-piper-de_DE-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-pavoque-low.html"><strong aria-hidden="true">13.12.</strong> vits-piper-de_DE-pavoque-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-ramona-low.html"><strong aria-hidden="true">13.13.</strong> vits-piper-de_DE-ramona-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-thorsten-high.html"><strong aria-hidden="true">13.14.</strong> vits-piper-de_DE-thorsten-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-thorsten-low.html"><strong aria-hidden="true">13.15.</strong> vits-piper-de_DE-thorsten-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-thorsten-medium.html"><strong aria-hidden="true">13.16.</strong> vits-piper-de_DE-thorsten-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="German/vits-piper-de_DE-thorsten_emotional-medium.html"><strong aria-hidden="true">13.17.</strong> vits-piper-de_DE-thorsten_emotional-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Greek/index.html"><strong aria-hidden="true">14.</strong> Greek</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Greek/vits-piper-el_GR-rapunzelina-low.html"><strong aria-hidden="true">14.1.</strong> vits-piper-el_GR-rapunzelina-low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hindi/index.html"><strong aria-hidden="true">15.</strong> Hindi</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hindi/vits-piper-hi_IN-pratham-medium.html"><strong aria-hidden="true">15.1.</strong> vits-piper-hi_IN-pratham-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hindi/vits-piper-hi_IN-priyamvada-medium.html"><strong aria-hidden="true">15.2.</strong> vits-piper-hi_IN-priyamvada-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hindi/vits-piper-hi_IN-rohan-medium.html"><strong aria-hidden="true">15.3.</strong> vits-piper-hi_IN-rohan-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hungarian/index.html"><strong aria-hidden="true">16.</strong> Hungarian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hungarian/vits-piper-hu_HU-anna-medium.html"><strong aria-hidden="true">16.1.</strong> vits-piper-hu_HU-anna-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hungarian/vits-piper-hu_HU-berta-medium.html"><strong aria-hidden="true">16.2.</strong> vits-piper-hu_HU-berta-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Hungarian/vits-piper-hu_HU-imre-medium.html"><strong aria-hidden="true">16.3.</strong> vits-piper-hu_HU-imre-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Icelandic/index.html"><strong aria-hidden="true">17.</strong> Icelandic</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Icelandic/vits-piper-is_IS-bui-medium.html"><strong aria-hidden="true">17.1.</strong> vits-piper-is_IS-bui-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Icelandic/vits-piper-is_IS-salka-medium.html"><strong aria-hidden="true">17.2.</strong> vits-piper-is_IS-salka-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Icelandic/vits-piper-is_IS-steinn-medium.html"><strong aria-hidden="true">17.3.</strong> vits-piper-is_IS-steinn-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Icelandic/vits-piper-is_IS-ugla-medium.html"><strong aria-hidden="true">17.4.</strong> vits-piper-is_IS-ugla-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Indonesian/index.html"><strong aria-hidden="true">18.</strong> Indonesian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Indonesian/vits-piper-id_ID-news_tts-medium.html"><strong aria-hidden="true">18.1.</strong> vits-piper-id_ID-news_tts-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Italian/index.html"><strong aria-hidden="true">19.</strong> Italian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Italian/vits-piper-it_IT-dii-high.html"><strong aria-hidden="true">19.1.</strong> vits-piper-it_IT-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Italian/vits-piper-it_IT-miro-high.html"><strong aria-hidden="true">19.2.</strong> vits-piper-it_IT-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Italian/vits-piper-it_IT-paola-medium.html"><strong aria-hidden="true">19.3.</strong> vits-piper-it_IT-paola-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Italian/vits-piper-it_IT-riccardo-x_low.html"><strong aria-hidden="true">19.4.</strong> vits-piper-it_IT-riccardo-x_low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Kazakh/index.html"><strong aria-hidden="true">20.</strong> Kazakh</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Kazakh/vits-piper-kk_KZ-iseke-x_low.html"><strong aria-hidden="true">20.1.</strong> vits-piper-kk_KZ-iseke-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Kazakh/vits-piper-kk_KZ-issai-high.html"><strong aria-hidden="true">20.2.</strong> vits-piper-kk_KZ-issai-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Kazakh/vits-piper-kk_KZ-raya-x_low.html"><strong aria-hidden="true">20.3.</strong> vits-piper-kk_KZ-raya-x_low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Latvian/index.html"><strong aria-hidden="true">21.</strong> Latvian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Latvian/vits-piper-lv_LV-aivars-medium.html"><strong aria-hidden="true">21.1.</strong> vits-piper-lv_LV-aivars-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Luxembourgish/index.html"><strong aria-hidden="true">22.</strong> Luxembourgish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Luxembourgish/vits-piper-lb_LU-marylux-medium.html"><strong aria-hidden="true">22.1.</strong> vits-piper-lb_LU-marylux-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Malayalam/index.html"><strong aria-hidden="true">23.</strong> Malayalam</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Malayalam/vits-piper-ml_IN-arjun-medium.html"><strong aria-hidden="true">23.1.</strong> vits-piper-ml_IN-arjun-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Malayalam/vits-piper-ml_IN-meera-medium.html"><strong aria-hidden="true">23.2.</strong> vits-piper-ml_IN-meera-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Nepali/index.html"><strong aria-hidden="true">24.</strong> Nepali</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Nepali/vits-piper-ne_NP-chitwan-medium.html"><strong aria-hidden="true">24.1.</strong> vits-piper-ne_NP-chitwan-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Nepali/vits-piper-ne_NP-google-medium.html"><strong aria-hidden="true">24.2.</strong> vits-piper-ne_NP-google-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Nepali/vits-piper-ne_NP-google-x_low.html"><strong aria-hidden="true">24.3.</strong> vits-piper-ne_NP-google-x_low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Norwegian/index.html"><strong aria-hidden="true">25.</strong> Norwegian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Norwegian/vits-piper-no_NO-talesyntese-medium.html"><strong aria-hidden="true">25.1.</strong> vits-piper-no_NO-talesyntese-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/index.html"><strong aria-hidden="true">26.</strong> Persian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/vits-piper-fa_IR-amir-medium.html"><strong aria-hidden="true">26.1.</strong> vits-piper-fa_IR-amir-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/vits-piper-fa_IR-ganji-medium.html"><strong aria-hidden="true">26.2.</strong> vits-piper-fa_IR-ganji-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/vits-piper-fa_IR-ganji_adabi-medium.html"><strong aria-hidden="true">26.3.</strong> vits-piper-fa_IR-ganji_adabi-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/vits-piper-fa_IR-gyro-medium.html"><strong aria-hidden="true">26.4.</strong> vits-piper-fa_IR-gyro-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Persian/vits-piper-fa_IR-reza_ibrahim-medium.html"><strong aria-hidden="true">26.5.</strong> vits-piper-fa_IR-reza_ibrahim-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/index.html"><strong aria-hidden="true">27.</strong> Polish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-darkman-medium.html"><strong aria-hidden="true">27.1.</strong> vits-piper-pl_PL-darkman-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-gosia-medium.html"><strong aria-hidden="true">27.2.</strong> vits-piper-pl_PL-gosia-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-jarvis_wg_glos-medium.html"><strong aria-hidden="true">27.3.</strong> vits-piper-pl_PL-jarvis_wg_glos-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-justyna_wg_glos-medium.html"><strong aria-hidden="true">27.4.</strong> vits-piper-pl_PL-justyna_wg_glos-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-mc_speech-medium.html"><strong aria-hidden="true">27.5.</strong> vits-piper-pl_PL-mc_speech-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-meski_wg_glos-medium.html"><strong aria-hidden="true">27.6.</strong> vits-piper-pl_PL-meski_wg_glos-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Polish/vits-piper-pl_PL-zenski_wg_glos-medium.html"><strong aria-hidden="true">27.7.</strong> vits-piper-pl_PL-zenski_wg_glos-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/index.html"><strong aria-hidden="true">28.</strong> Portuguese</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-cadu-medium.html"><strong aria-hidden="true">28.1.</strong> vits-piper-pt_BR-cadu-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-dii-high.html"><strong aria-hidden="true">28.2.</strong> vits-piper-pt_BR-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-edresson-low.html"><strong aria-hidden="true">28.3.</strong> vits-piper-pt_BR-edresson-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-faber-medium.html"><strong aria-hidden="true">28.4.</strong> vits-piper-pt_BR-faber-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-jeff-medium.html"><strong aria-hidden="true">28.5.</strong> vits-piper-pt_BR-jeff-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_BR-miro-high.html"><strong aria-hidden="true">28.6.</strong> vits-piper-pt_BR-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_PT-dii-high.html"><strong aria-hidden="true">28.7.</strong> vits-piper-pt_PT-dii-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_PT-miro-high.html"><strong aria-hidden="true">28.8.</strong> vits-piper-pt_PT-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Portuguese/vits-piper-pt_PT-tugao-medium.html"><strong aria-hidden="true">28.9.</strong> vits-piper-pt_PT-tugao-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Romanian/index.html"><strong aria-hidden="true">29.</strong> Romanian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Romanian/vits-piper-ro_RO-mihai-medium.html"><strong aria-hidden="true">29.1.</strong> vits-piper-ro_RO-mihai-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Russian/index.html"><strong aria-hidden="true">30.</strong> Russian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Russian/vits-piper-ru_RU-denis-medium.html"><strong aria-hidden="true">30.1.</strong> vits-piper-ru_RU-denis-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Russian/vits-piper-ru_RU-dmitri-medium.html"><strong aria-hidden="true">30.2.</strong> vits-piper-ru_RU-dmitri-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Russian/vits-piper-ru_RU-irina-medium.html"><strong aria-hidden="true">30.3.</strong> vits-piper-ru_RU-irina-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Russian/vits-piper-ru_RU-ruslan-medium.html"><strong aria-hidden="true">30.4.</strong> vits-piper-ru_RU-ruslan-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Serbian/index.html"><strong aria-hidden="true">31.</strong> Serbian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Serbian/vits-piper-sr_RS-serbski_institut-medium.html"><strong aria-hidden="true">31.1.</strong> vits-piper-sr_RS-serbski_institut-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Slovak/index.html"><strong aria-hidden="true">32.</strong> Slovak</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Slovak/vits-piper-sk_SK-lili-medium.html"><strong aria-hidden="true">32.1.</strong> vits-piper-sk_SK-lili-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Slovenian/index.html"><strong aria-hidden="true">33.</strong> Slovenian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Slovenian/vits-piper-sl_SI-artur-medium.html"><strong aria-hidden="true">33.1.</strong> vits-piper-sl_SI-artur-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/index.html"><strong aria-hidden="true">34.</strong> Spanish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_AR-daniela-high.html"><strong aria-hidden="true">34.1.</strong> vits-piper-es_AR-daniela-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_ES-carlfm-x_low.html"><strong aria-hidden="true">34.2.</strong> vits-piper-es_ES-carlfm-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_ES-davefx-medium.html"><strong aria-hidden="true">34.3.</strong> vits-piper-es_ES-davefx-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_ES-glados-medium.html"><strong aria-hidden="true">34.4.</strong> vits-piper-es_ES-glados-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_ES-miro-high.html"><strong aria-hidden="true">34.5.</strong> vits-piper-es_ES-miro-high</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_ES-sharvard-medium.html"><strong aria-hidden="true">34.6.</strong> vits-piper-es_ES-sharvard-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_MX-ald-medium.html"><strong aria-hidden="true">34.7.</strong> vits-piper-es_MX-ald-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Spanish/vits-piper-es_MX-claude-high.html"><strong aria-hidden="true">34.8.</strong> vits-piper-es_MX-claude-high</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Swahili/index.html"><strong aria-hidden="true">35.</strong> Swahili</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Swahili/vits-piper-sw_CD-lanfrica-medium.html"><strong aria-hidden="true">35.1.</strong> vits-piper-sw_CD-lanfrica-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Swedish/index.html"><strong aria-hidden="true">36.</strong> Swedish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Swedish/vits-piper-sv_SE-lisa-medium.html"><strong aria-hidden="true">36.1.</strong> vits-piper-sv_SE-lisa-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Swedish/vits-piper-sv_SE-nst-medium.html"><strong aria-hidden="true">36.2.</strong> vits-piper-sv_SE-nst-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Turkish/index.html"><strong aria-hidden="true">37.</strong> Turkish</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Turkish/vits-piper-tr_TR-dfki-medium.html"><strong aria-hidden="true">37.1.</strong> vits-piper-tr_TR-dfki-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Turkish/vits-piper-tr_TR-fahrettin-medium.html"><strong aria-hidden="true">37.2.</strong> vits-piper-tr_TR-fahrettin-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Turkish/vits-piper-tr_TR-fettah-medium.html"><strong aria-hidden="true">37.3.</strong> vits-piper-tr_TR-fettah-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Ukrainian/index.html"><strong aria-hidden="true">38.</strong> Ukrainian</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Ukrainian/vits-piper-uk_UA-lada-x_low.html"><strong aria-hidden="true">38.1.</strong> vits-piper-uk_UA-lada-x_low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Ukrainian/vits-piper-uk_UA-ukrainian_tts-medium.html"><strong aria-hidden="true">38.2.</strong> vits-piper-uk_UA-ukrainian_tts-medium</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Vietnamese/index.html"><strong aria-hidden="true">39.</strong> Vietnamese</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Vietnamese/vits-piper-vi_VN-25hours_single-low.html"><strong aria-hidden="true">39.1.</strong> vits-piper-vi_VN-25hours_single-low</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Vietnamese/vits-piper-vi_VN-vais1000-medium.html"><strong aria-hidden="true">39.2.</strong> vits-piper-vi_VN-vais1000-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Vietnamese/vits-piper-vi_VN-vivos-x_low.html"><strong aria-hidden="true">39.3.</strong> vits-piper-vi_VN-vivos-x_low</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Welsh/index.html"><strong aria-hidden="true">40.</strong> Welsh</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Welsh/vits-piper-cy_GB-bu_tts-medium.html"><strong aria-hidden="true">40.1.</strong> vits-piper-cy_GB-bu_tts-medium</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Welsh/vits-piper-cy_GB-gwryw_gogleddol-medium.html"><strong aria-hidden="true">40.2.</strong> vits-piper-cy_GB-gwryw_gogleddol-medium</a></span></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

