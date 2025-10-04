<template>
  <div>
    <v-alert v-if="!card && !loading" color="muted" variant="text">
      No imported card data found for this character.
    </v-alert>

    <div v-else>
      <v-toolbar flat density="compact">
        <v-toolbar-title class="text-subtitle-2">Imported Card Data</v-toolbar-title>
        <v-spacer></v-spacer>
        <v-btn variant="text" color="primary" @click="copyJSON" prepend-icon="mdi-content-copy">Copy JSON</v-btn>
        <v-btn variant="text" color="secondary" @click="reload" prepend-icon="mdi-refresh">Reload</v-btn>
      </v-toolbar>
      <v-divider></v-divider>
      <v-card-text class="pa-3">
        <div v-if="loading">
          <v-progress-linear indeterminate color="primary"></v-progress-linear>
        </div>
        <div v-else class="json-pre">
          <pre>{{ prettyJSON }}</pre>
          <div class="mt-2 text-caption text-grey" v-if="filename">Source: {{ filename }}</div>
        </div>
      </v-card-text>
    </div>
  </div>
</template>

<script>
export default {
  name: 'WorldStateManagerCharacterCard',
  props: {
    immutableCharacter: Object,
  },
  inject: ['getWebsocket', 'registerMessageHandler', 'unregisterMessageHandler'],
  data() {
    return {
      loading: false,
      card: null,
      filename: null,
    }
  },
  computed: {
    prettyJSON() {
      return this.card ? JSON.stringify(this.card, null, 2) : '';
    }
  },
  watch: {
    immutableCharacter: {
      handler(n) {
        if (n && n.name) {
          this.load();
        } else {
          this.card = null;
        }
      },
      immediate: true,
      deep: false,
    }
  },
  methods: {
    reload() {
      this.load();
    },
    load() {
      if (!this.immutableCharacter || !this.immutableCharacter.name) return;
      this.loading = true;
      this.getWebsocket().send(JSON.stringify({
        type: 'world_state_manager',
        action: 'get_character_card',
        name: this.immutableCharacter.name,
      }));
    },
    copyJSON() {
      if (!this.card) return;
      navigator.clipboard.writeText(JSON.stringify(this.card, null, 2));
    },
    handleMessage(message) {
      if (message.type !== 'world_state_manager') return;
      if (message.action === 'character_card') {
        if (!message.data || !this.immutableCharacter) return;
        if (message.data.name !== this.immutableCharacter.name) return;
        this.card = message.data.card;
        this.filename = message.data.filename || null;
        this.loading = false;
      }
    }
  },
  mounted() {
    this.registerMessageHandler(this.handleMessage);
  },
  unmounted() {
    this.unregisterMessageHandler(this.handleMessage);
  }
}
</script>

<style scoped>
.json-pre {
  background: #0e0e0e0f;
  border-radius: 6px;
  padding: 8px;
  overflow: auto;
}
pre { white-space: pre-wrap; }
</style>
