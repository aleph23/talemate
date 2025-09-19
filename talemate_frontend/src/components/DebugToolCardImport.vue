<template>
  <div>
    <v-card variant="text">
      <v-card-title>
        <v-icon class="mr-2" color="primary">mdi-book-information-variant</v-icon>
        Card Import (Lorebook)
        <v-spacer></v-spacer>
        <v-btn size="small" variant="tonal" color="primary" @click="refresh" :disabled="busy" prepend-icon="mdi-refresh">Refresh</v-btn>
      </v-card-title>
      <v-card-text>
        <div v-if="!cardImport">
          <v-alert type="info" variant="tonal">No card import data available for the current scene.</v-alert>
        </div>
        <div v-else>
          <div class="mb-2 text-caption">Spec: <strong>{{ cardImport.spec }}</strong> — Entries: <strong>{{ cardImport.lorebook_entries_count }}</strong> — Character: <strong>{{ cardImport.card_name }}</strong></div>
          <Codemirror v-model="jsonText" :extensions="extensions" :style="editorStyle"></Codemirror>
        </div>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
import { Codemirror } from 'vue-codemirror'
import { json } from '@codemirror/lang-json'
import { oneDark } from '@codemirror/theme-one-dark'
import { EditorView } from '@codemirror/view'

export default {
  name: 'DebugToolCardImport',
  components: { Codemirror },
  data() {
    return {
      cardImport: null,
      jsonText: '{}',
      busy: false,
    }
  },
  inject: ['getWebsocket', 'registerMessageHandler', 'unregisterMessageHandler'],
  methods: {
    refresh() {
      this.getSceneState()
    },
    getSceneState() {
      this.busy = true
      this.getWebsocket().send(JSON.stringify({
        type: 'devtools',
        action: 'get_scene_state',
      }))
    },
    handleMessage(data) {
      if (data.type !== 'devtools') return
      if (data.action === 'scene_state') {
        this.busy = false
        const state = data.data
        this.cardImport = (state.agent_state && state.agent_state.card_import) ? state.agent_state.card_import : null
        this.jsonText = JSON.stringify(this.cardImport || {}, null, 2)
      }
    }
  },
  created() {
    this.registerMessageHandler(this.handleMessage)
    this.refresh()
  },
  unmounted() {
    this.unregisterMessageHandler(this.handleMessage)
  },
  setup() {
    const extensions = [ json(), oneDark, EditorView.lineWrapping ]
    const editorStyle = { maxHeight: '560px', minHeight: '320px' }
    return { extensions, editorStyle }
  }
}
</script>

<style scoped>
</style>
